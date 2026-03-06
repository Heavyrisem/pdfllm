import statistics

import fitz  # PyMuPDF
import pdfplumber

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

_MAX_RECURSION_DEPTH = 3
_OUTLIER_AREA_MULTIPLIER = 4.0
_BBOX_ROUND_PRECISION = 1
_BG_THRESHOLD = 0.7      # _collect_drawing_rects: 배경 프레임 판단 비율
_LOCAL_BG_RATIO = 0.8    # band 높이 대비 로컬 배경 판단 비율


# ---------------------------------------------------------------------------
# 내부 유틸리티
# ---------------------------------------------------------------------------


def _extract_text_clip(pdf_path: str, clip: fitz.Rect, page_idx: int) -> str:
    """pdfplumber로 clip 영역의 텍스트 추출. 실패 시 PyMuPDF fallback."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_idx]
            cropped = page.crop((clip.x0, clip.y0, clip.x1, clip.y1))
            text = cropped.extract_text() or ""
        if text.strip():
            return text
    except Exception:
        pass

    # fallback: PyMuPDF (pdfplumber 실패 또는 빈 결과 시에만 실행)
    try:
        doc = fitz.open(pdf_path)
        text = doc[page_idx].get_text("text", clip=clip)
        doc.close()
        return text or ""
    except Exception:
        return ""


def _apply_padding(
    bbox: tuple,
    padding_pt: float,
    page_w: float,
    page_h: float,
) -> tuple:
    """bbox에 padding을 적용하고 페이지 경계로 클램핑한다."""
    x0, y0, x1, y1 = bbox
    return (
        max(0.0, x0 - padding_pt),
        max(0.0, y0 - padding_pt),
        min(page_w, x1 + padding_pt),
        min(page_h, y1 + padding_pt),
    )


def get_page_count(pdf_path: str) -> dict:
    """PDF 페이지 수와 각 페이지 크기(pt 단위)를 반환."""
    doc = fitz.open(pdf_path)
    pages = [
        {"width": doc[i].rect.width, "height": doc[i].rect.height}
        for i in range(len(doc))
    ]
    doc.close()
    return {"page_count": len(pages), "pages": pages}


# ── detect_layout / get_region 관련 함수 ──────────────────────────────────────

def _collect_text_rects(page: fitz.Page) -> list[tuple[float, float, float, float]]:
    """텍스트 블록 bbox 수집 (cut line 계산 전용)."""
    rects = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1 = block[0], block[1], block[2], block[3]
        if (x1 - x0) > 2 and (y1 - y0) > 2:
            rects.append((x0, y0, x1, y1))
    return rects


def _collect_drawing_rects(
    page: fitz.Page,
    bg_threshold: float = _BG_THRESHOLD,
) -> list[tuple[float, float, float, float]]:
    """
    드로잉 bbox 수집.
    페이지 전체를 덮는 배경 프레임(bg_threshold 이상)은 제외한다.
    """
    page_w = page.rect.width
    page_h = page.rect.height
    max_draw_w = page_w * bg_threshold
    max_draw_h = page_h * bg_threshold

    rects = []
    for d in page.get_drawings():
        r = d.get("rect")
        if r is None:
            continue
        x0, y0, x1, y1 = r.x0, r.y0, r.x1, r.y1
        w, h = x1 - x0, y1 - y0
        if w < 5 or h < 5:
            continue
        if w >= max_draw_w and h >= max_draw_h:
            continue
        rects.append((x0, y0, x1, y1))
    return rects


def _find_axis_cuts(
    rects: list[tuple],
    axis: int,
    axis_max: float,
    min_gap: float,
    axis_min: float = 0.0,
) -> list[float]:
    """
    1D sweep으로 whitespace gap의 중점을 cut line으로 반환한다.

    axis=0: X축 기준 (수직 절단선)
    axis=1: Y축 기준 (수평 절단선)
    axis_min/axis_max: 분석 범위 (서브 영역 재귀 분석 시 사용)
    """
    if not rects:
        return []

    events: list[tuple[float, str]] = []
    for r in rects:
        # 좌표를 axis 범위로 클램핑 (서브 영역 경계에 걸친 rect 처리)
        lo = max(r[axis], axis_min)
        hi = min(r[axis + 2], axis_max)
        if lo >= hi:
            continue
        events.append((lo, "open"))
        events.append((hi, "close"))

    if not events:
        return []

    events.sort(key=lambda e: (e[0], 0 if e[1] == "open" else 1))

    cuts = []
    depth = 0
    gap_start = axis_min

    for coord, kind in events:
        if kind == "open":
            if depth == 0:
                gap_end = coord
                if gap_start < gap_end and (gap_end - gap_start) >= min_gap:
                    cuts.append((gap_start + gap_end) / 2)
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                gap_start = coord

    if depth == 0 and gap_start < axis_max and (axis_max - gap_start) >= min_gap:
        cuts.append((gap_start + axis_max) / 2)

    return cuts


def _detect_regions_in_area(
    text_rects: list[tuple],
    draw_rects: list[tuple],
    all_rects: list[tuple],
    area_x0: float,
    area_y0: float,
    area_x1: float,
    area_y1: float,
    min_gap_pt: float,
    padding_pt: float,
    page_w: float,
    page_h: float,
) -> list[tuple]:
    """
    주어진 area 내에서 region bbox 목록을 반환한다.

    cut line 전략:
    - Y-cuts: 텍스트 + 높이 >= thin_pt 인 드로잉만 사용
              → 얇은 connector/border rect(bridge)가 Y 방향 gap을 막는 문제 방지
    - X-cuts: 텍스트 + (band 높이 80% 미만) + (너비 >= thin_pt) 드로잉
              → 방안 A(로컬 배경) + thin-rect 이중 필터
    - tight bbox: all_rects(텍스트+드로잉 전체) 기준

    thin_pt = min_gap_pt * 3 (기본 30pt × 3 = 90pt)
    """
    thin_pt = min_gap_pt * 3

    def _in_area(r: tuple) -> bool:
        return r[0] < area_x1 and r[2] > area_x0 and r[1] < area_y1 and r[3] > area_y0

    def _x_overlap(r: tuple) -> bool:
        """X 범위만 검사 (Y-cut 계산에서 현재 X 범위 내 rects만 사용)."""
        return r[0] < area_x1 and r[2] > area_x0

    area_text = [r for r in text_rects if _in_area(r)]
    area_draw = [r for r in draw_rects if _in_area(r)]
    area_all  = [r for r in all_rects  if _in_area(r)]

    if not area_text and not area_draw:
        return []

    # Y-cuts용 rects: 텍스트 전체 + 충분히 높은 드로잉
    # 단, 현재 X 범위에 걸치는 rects만 사용
    # → 다른 X 컬럼의 tall rect가 이 영역의 Y-gap을 막는 문제 방지
    text_x = [r for r in text_rects if _x_overlap(r) and r[1] < area_y1 and r[3] > area_y0]
    draw_x = [r for r in draw_rects if _x_overlap(r) and r[1] < area_y1 and r[3] > area_y0]
    draw_tall = [r for r in draw_x if (r[3] - r[1]) >= thin_pt]

    cut_y = text_x + draw_tall
    if not cut_y:
        cut_y = text_x + draw_x  # fallback: 얇은 것밖에 없으면 전부 사용
    if not cut_y:
        return []

    # ① Y축 수평 절단선
    h_cuts = _find_axis_cuts(cut_y, axis=1, axis_max=area_y1, min_gap=min_gap_pt, axis_min=area_y0)
    y_boundaries = [area_y0] + h_cuts + [area_y1]
    bands = [(y_boundaries[i], y_boundaries[i + 1]) for i in range(len(y_boundaries) - 1)]

    result: list[tuple] = []

    for band_y0, band_y1 in bands:
        band_text = [r for r in area_text if r[1] < band_y1 and r[3] > band_y0]
        band_draw = [r for r in area_draw if r[1] < band_y1 and r[3] > band_y0]

        if not band_text and not band_draw:
            continue

        # ② X축 수직 절단선
        # 방안 A: band 높이의 _LOCAL_BG_RATIO 이상 rect → 로컬 배경 제거
        band_h = band_y1 - band_y0
        local_bg_h = band_h * _LOCAL_BG_RATIO
        # thin-rect 필터: 너비 >= thin_pt인 드로잉만 X-cut에 사용
        band_draw_x = [r for r in band_draw if (r[3] - r[1]) < local_bg_h and (r[2] - r[0]) >= thin_pt]

        cut_x = band_text + band_draw_x
        if not cut_x:
            # fallback: 필터 후 비어있으면 로컬 배경 필터만 적용
            cut_x = band_text + [r for r in band_draw if (r[3] - r[1]) < local_bg_h]
        if not cut_x:
            cut_x = band_text + band_draw

        v_cuts = _find_axis_cuts(cut_x, axis=0, axis_max=area_x1, min_gap=min_gap_pt, axis_min=area_x0)
        x_boundaries = [area_x0] + v_cuts + [area_x1]
        col_ranges = [(x_boundaries[i], x_boundaries[i + 1]) for i in range(len(x_boundaries) - 1)]

        for col_x0, col_x1 in col_ranges:
            col_cut = [r for r in band_text + band_draw if r[0] < col_x1 and r[2] > col_x0]
            if not col_cut:
                continue

            col_all = [
                r for r in area_all
                if r[0] < col_x1 and r[2] > col_x0 and r[1] < band_y1 and r[3] > band_y0
            ]
            source = col_all if col_all else col_cut

            tx0 = min(r[0] for r in source)
            ty0 = min(r[1] for r in source)
            tx1 = max(r[2] for r in source)
            ty1 = max(r[3] for r in source)

            result.append(_apply_padding((tx0, ty0, tx1, ty1), padding_pt, page_w, page_h))

    return result


def _recursive_split(
    boxes: list[tuple],
    current_gap: float,
    depth: int,
    *,
    eff_text: list[tuple],
    eff_draw: list[tuple],
    all_rects: list[tuple],
    padding_pt: float,
    page_w: float,
    page_h: float,
) -> list[tuple]:
    """너무 큰 region을 current_gap/2로 재귀 분할한다 (최대 _MAX_RECURSION_DEPTH 단계)."""
    if depth >= _MAX_RECURSION_DEPTH or current_gap / 2 < 5.0 or len(boxes) < 2:
        return boxes

    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    median_area = statistics.median(areas)
    threshold = median_area * _OUTLIER_AREA_MULTIPLIER
    next_gap = current_gap / 2

    result: list[tuple] = []
    for bbox, area in zip(boxes, areas):
        if area > threshold:
            sub = _detect_regions_in_area(
                eff_text, eff_draw, all_rects,
                bbox[0], bbox[1], bbox[2], bbox[3],
                next_gap, padding_pt, page_w, page_h,
            )
            if len(sub) > 1:
                result.extend(_recursive_split(
                    sub, next_gap, depth + 1,
                    eff_text=eff_text, eff_draw=eff_draw, all_rects=all_rects,
                    padding_pt=padding_pt, page_w=page_w, page_h=page_h,
                ))
                continue
        result.append(bbox)
    return result


def detect_content_regions(
    pdf_path: str,
    page_idx: int = 0,
    min_gap_pt: float = 30.0,
    padding_pt: float = 6.0,
) -> list[dict]:
    """
    PyMuPDF 기반 whitespace 분석으로 콘텐츠 region 목록을 반환한다.

    개선 사항:
    - 방안 A: X-cuts 시 band 높이 80% 이상 rect를 로컬 배경으로 제거
    - thin-rect 필터: Y-cuts에 높이 < min_gap*3, X-cuts에 너비 < min_gap*3인 드로잉 제외
                     → connector/border 요소가 whitespace gap을 막는 문제 방지
    - 방안 C: 너무 큰 region(중앙값 면적 4배 이상)을 min_gap/2로 재귀 분할
    - fallback: text_rects가 없으면 drawing_rects를 text_rects 역할로 사용
    """
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    page_w = page.rect.width
    page_h = page.rect.height

    text_rects = _collect_text_rects(page)
    draw_rects = _collect_drawing_rects(page)
    doc.close()

    all_rects = text_rects + draw_rects

    if not all_rects:
        return []

    # fallback: 텍스트 없는 PDF (이미지/드로잉 전용)
    # drawing_rects를 tall(실질 콘텐츠)과 short(장식/선)로 분리하여
    # tall → text_rects 역할, short → draw_rects 역할로 사용
    # → thin_pt 필터가 draw_rects에만 적용되는 구조를 그대로 활용
    if text_rects:
        eff_text = text_rects
        eff_draw = draw_rects
    else:
        thin_threshold = min_gap_pt * 3
        eff_text = [r for r in draw_rects if (r[3] - r[1]) >= thin_threshold]
        eff_draw = [r for r in draw_rects if (r[3] - r[1]) < thin_threshold]
        if not eff_text:  # 모두 얇은 경우 전체 사용
            eff_text = draw_rects
            eff_draw = []

    # ① 1차 탐지
    bboxes = _detect_regions_in_area(
        eff_text, eff_draw, all_rects,
        0.0, 0.0, page_w, page_h,
        min_gap_pt, padding_pt, page_w, page_h,
    )

    if not bboxes:
        return []

    # ② 재귀 분할 (방안 C)
    final_bboxes = _recursive_split(
        bboxes, min_gap_pt, depth=0,
        eff_text=eff_text, eff_draw=eff_draw, all_rects=all_rects,
        padding_pt=padding_pt, page_w=page_w, page_h=page_h,
    )

    # 중복 bbox 제거 (순서 유지)
    seen: set[tuple] = set()
    unique_bboxes: list[tuple] = []
    for b in final_bboxes:
        key = (
            round(b[0], _BBOX_ROUND_PRECISION),
            round(b[1], _BBOX_ROUND_PRECISION),
            round(b[2], _BBOX_ROUND_PRECISION),
            round(b[3], _BBOX_ROUND_PRECISION),
        )
        if key not in seen:
            seen.add(key)
            unique_bboxes.append(b)

    return [{"idx": i, "bbox_pt": list(b)} for i, b in enumerate(unique_bboxes)]


def render_region(
    pdf_path: str,
    page_idx: int,
    bbox_pt: list[float],
    dpi: int = 72,
    padding_pt: float = 6.0,
) -> bytes:
    """
    bbox_pt([x0, y0, x1, y1])로 지정한 영역을 JPEG bytes로 렌더링한다.
    padding_pt만큼 bbox를 확장하고 page 경계로 클램핑한다.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    page_w = page.rect.width
    page_h = page.rect.height

    clip = fitz.Rect(*_apply_padding(tuple(bbox_pt), padding_pt, page_w, page_h))

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    img_bytes = pix.tobytes("jpeg")

    del pix
    fitz.TOOLS.store_shrink(100)
    doc.close()

    return img_bytes
