import shutil
import subprocess
import tempfile
from math import ceil
from pathlib import Path

import fitz  # PyMuPDF


def _round_grid(n: int) -> int:
    """후보 목록 중 n 이상인 최솟값 반환."""
    candidates = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
    for c in candidates:
        if c >= n:
            return c
    return candidates[-1]


def _calc_cell_clip(
    page_w: float,
    page_h: float,
    cell_idx: int,
    rows: int,
    cols: int,
    overlap: float = 0.0,
) -> tuple[fitz.Rect, float, float, int, int]:
    """
    셀 인덱스로부터 클립 영역(fitz.Rect)과 메타데이터를 계산한다.

    Returns:
        (clip, tile_w, tile_h, row, col)
    """
    row, col = divmod(cell_idx, cols)
    tile_w = page_w / cols
    tile_h = page_h / rows

    pad_x = overlap * tile_w
    pad_y = overlap * tile_h

    x0 = max(0.0, col * tile_w - pad_x)
    y0 = max(0.0, row * tile_h - pad_y)
    x1 = min(page_w, (col + 1) * tile_w + pad_x)
    y1 = min(page_h, (row + 1) * tile_h + pad_y)

    return fitz.Rect(x0, y0, x1, y1), tile_w, tile_h, row, col


def analyze_page(
    pdf_path: str,
    page_idx: int = 0,
    target_tile_pt: int = 1500,
) -> dict:
    """
    페이지 크기·텍스트 밀도를 분석하고 적절한 그리드 크기를 추천한다.

    Args:
        pdf_path: PDF 파일 절대 경로
        page_idx: 분석할 페이지 번호 (0부터 시작)
        target_tile_pt: 목표 타일 크기 (포인트 단위, 기본 1500)

    Returns:
        page_size_pt, page_size_mm, text_block_count, total_chars,
        image_block_count, text_density, suggested_grid, suggested_tile_size_pt
    """
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    rect = page.rect
    page_w = rect.width
    page_h = rect.height

    text_dict = page.get_text("dict")
    blocks = text_dict.get("blocks", [])

    text_blocks = [b for b in blocks if b.get("type") == 0]
    image_blocks = [b for b in blocks if b.get("type") == 1]

    total_chars = sum(
        len(span.get("text", ""))
        for block in text_blocks
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    )

    doc.close()

    area = page_w * page_h
    density = total_chars / area if area > 0 else 0.0

    base_cols = ceil(page_w / target_tile_pt)
    base_rows = ceil(page_h / target_tile_pt)

    if density > 0.01:
        base_cols = ceil(base_cols * 1.5)
        base_rows = ceil(base_rows * 1.5)

    suggested_rows = _round_grid(base_rows)
    suggested_cols = _round_grid(base_cols)

    PT_TO_MM = 0.3528
    return {
        "page_size_pt": {"width": page_w, "height": page_h},
        "page_size_mm": {"width": round(page_w * PT_TO_MM, 1), "height": round(page_h * PT_TO_MM, 1)},
        "text_block_count": len(text_blocks),
        "total_chars": total_chars,
        "image_block_count": len(image_blocks),
        "text_density": density,
        "suggested_grid": {"rows": suggested_rows, "cols": suggested_cols},
        "suggested_tile_size_pt": {
            "width": round(page_w / suggested_cols, 1),
            "height": round(page_h / suggested_rows, 1),
        },
    }


def get_page_count(pdf_path: str) -> dict:
    """PDF 페이지 수와 각 페이지 크기(pt 단위)를 반환."""
    doc = fitz.open(pdf_path)
    pages = [
        {"width": doc[i].rect.width, "height": doc[i].rect.height}
        for i in range(len(doc))
    ]
    doc.close()
    return {"page_count": len(pages), "pages": pages}


def render_overview(pdf_path: str, max_px: int = 2048, page_idx: int = 0) -> bytes:
    """전체 페이지를 max_px 이내로 축소 렌더링하여 JPEG bytes 반환."""
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    rect = page.rect
    scale = min(max_px / rect.width, max_px / rect.height)
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("jpeg")
    del pix
    doc.close()
    return img_bytes


def render_tile(
    pdf_path: str,
    cell_idx: int,
    rows: int,
    cols: int,
    dpi: int = 72,
    page_idx: int = 0,
    overlap: float = 0.0,
) -> bytes:
    """
    그리드 셀 인덱스에 해당하는 영역을 고해상도로 렌더링하여 JPEG bytes 반환.
    메모리 누수 방지를 위해 처리 후 즉시 해제.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    rect = page.rect

    clip, _, _, _, _ = _calc_cell_clip(rect.width, rect.height, cell_idx, rows, cols, overlap)

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    img_bytes = pix.tobytes("jpeg")

    del pix
    fitz.TOOLS.store_shrink(100)
    doc.close()

    return img_bytes


def render_tile_as_pdf(
    pdf_path: str,
    cell_idx: int,
    rows: int,
    cols: int,
    page_idx: int = 0,
    overlap: float = 0.0,
) -> bytes:
    """
    그리드 셀 영역을 Ghostscript로 최적화하여 PDF bytes 반환.

    PyMuPDF로 cropbox를 설정한 중간 PDF를 생성한 뒤 Ghostscript로
    재처리하여 해당 영역 밖의 이미지/폰트를 제거합니다.
    시스템에 gs(Ghostscript)가 설치되어 있어야 합니다.
    """
    gs = shutil.which("gs")
    if not gs:
        raise RuntimeError("Ghostscript(gs)가 설치되어 있지 않습니다. brew install ghostscript")

    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    rect = page.rect

    clip, _, _, _, _ = _calc_cell_clip(rect.width, rect.height, cell_idx, rows, cols, overlap)

    with tempfile.TemporaryDirectory() as tmp:
        # 1단계: PyMuPDF로 cropbox 설정된 중간 PDF 생성
        mid_path = Path(tmp) / "mid.pdf"
        mid_doc = fitz.open()
        mid_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
        mid_doc[0].set_cropbox(clip)
        mid_doc.save(str(mid_path))
        mid_doc.close()
        doc.close()

        # 2단계: Ghostscript로 재처리 (미사용 리소스 제거 + 폰트 서브셋)
        out_path = Path(tmp) / "out.pdf"
        try:
            subprocess.run(
                [
                    gs,
                    "-sDEVICE=pdfwrite",
                    "-dNOPAUSE", "-dBATCH", "-dQUIET",
                    "-dSubsetFonts=true",
                    "-dEmbedAllFonts=true",
                    "-dCompressFonts=true",
                    "-dDownsampleColorImages=false",
                    "-dDownsampleGrayImages=false",
                    "-dDownsampleMonoImages=false",
                    f"-sOutputFile={out_path}",
                    str(mid_path),
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="replace").strip()
            raise RuntimeError(
                f"Ghostscript가 셀 {cell_idx} 처리에 실패했습니다.\n"
                f"원인: {stderr or '알 수 없는 오류'}\n"
                "대안: get_tile 도구를 사용하면 Ghostscript 없이 이미지로 추출할 수 있습니다."
            ) from None

        return out_path.read_bytes()


def get_page_structure(
    pdf_path: str,
    rows: int,
    cols: int,
    page_idx: int = 0,
    overlap: float = 0.0,
    include_empty: bool = False,
) -> dict:
    """
    셀별 텍스트/이미지 유무와 PDF 목차(TOC)를 반환한다.

    각 셀에 대해 has_text, has_image, text_preview를 계산하여
    LLM이 불필요한 get_tile 호출을 최소화할 수 있도록 한다.

    overlap을 지정하면 get_tile/get_tile_text와 동일한 경계 확장 후 검사하므로
    경계 근처 콘텐츠를 놓치지 않는다.

    주의: toc의 page 필드는 PyMuPDF get_toc() 기준으로 1-indexed이다.
    page_idx 파라미터(0-indexed)와 1 차이가 있으므로, TOC page 값을
    page_idx로 사용하려면 1을 빼야 한다.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    page_w = page.rect.width
    page_h = page.rect.height

    blocks = page.get_text("dict")["blocks"]
    toc = doc.get_toc()

    cells = {}
    for cell_idx in range(rows * cols):
        cell_rect, _, _, _, _ = _calc_cell_clip(page_w, page_h, cell_idx, rows, cols, overlap)

        has_text = False
        has_image = False
        text_preview = ""

        for block in blocks:
            block_rect = fitz.Rect(block["bbox"])
            if not cell_rect.intersects(block_rect):
                continue
            if block["type"] == 0:  # 텍스트
                has_text = True
                if not text_preview:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text_preview += span.get("text", "")
                            if len(text_preview) >= 80:
                                break
                        if len(text_preview) >= 80:
                            break
            elif block["type"] == 1:  # 이미지
                has_image = True

        cells[cell_idx] = {
            "has_text": has_text,
            "has_image": has_image,
            "text_preview": text_preview[:80].strip(),
        }

    if not include_empty:
        cells = {k: v for k, v in cells.items() if v["has_text"] or v["has_image"]}

    doc.close()
    return {"cells": cells, "toc": toc}


def extract_tile_text(
    pdf_path: str,
    cell_idx: int,
    rows: int,
    cols: int,
    page_idx: int = 0,
    overlap: float = 0.0,
    format: str = "text",
) -> dict:
    """
    그리드 셀 인덱스에 해당하는 영역의 텍스트를 추출하여 반환.
    render_tile과 동일한 clip 계산 로직을 사용하므로 좌표가 일치합니다.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    rect = page.rect

    clip, _, _, _, _ = _calc_cell_clip(rect.width, rect.height, cell_idx, rows, cols, overlap)
    text_dict = page.get_text("dict", clip=clip)
    doc.close()

    # type=0: 텍스트 블록, type=1: 이미지 블록 (bytes 포함으로 JSON 직렬화 불가)
    text_blocks = [b for b in text_dict.get("blocks", []) if b.get("type") == 0]

    if format == "text":
        lines_text = []
        for block in text_blocks:
            for line in block.get("lines", []):
                line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                if line_text.strip():
                    lines_text.append(line_text)
        return {
            "text": "\n".join(lines_text),
            "cell_idx": cell_idx,
            "clip_rect": [clip.x0, clip.y0, clip.x1, clip.y1],
        }

    # format == "compact": bbox + text만 유지
    compact_blocks = []
    for block in text_blocks:
        compact_lines = []
        for line in block.get("lines", []):
            compact_spans = [
                {"text": s.get("text", ""), "bbox": s["bbox"]}
                for s in line.get("spans", [])
            ]
            compact_lines.append({"bbox": line["bbox"], "spans": compact_spans})
        compact_blocks.append({"bbox": block["bbox"], "lines": compact_lines})

    return {
        "blocks": compact_blocks,
        "cell_idx": cell_idx,
        "clip_rect": [clip.x0, clip.y0, clip.x1, clip.y1],
    }
