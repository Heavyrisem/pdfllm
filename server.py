import json
import os
import tempfile

from mcp.server.fastmcp import FastMCP, Image

from pdf_tiler import render_overview, render_tile, render_tile_as_pdf, get_page_count as _get_page_count, extract_tile_text, analyze_page as _analyze_page, get_page_structure as _get_page_structure, search_cells as _search_cells
from scaffold import add_grid_overlay
from monitor import MonitorServer, track_call

mcp = FastMCP(
    "pdfllm",
    instructions="""이 서버는 LLM이 직접 읽기 어려운 대용량 PDF(수십~수백 MB, 수억 픽셀 규모)를
효율적으로 분석하기 위한 타일링 도구를 제공합니다.

## 권장 워크플로우

1. **get_page_count** → 페이지 수·크기 파악
2. **suggest_grid** → 적절한 grid_rows/grid_cols 추천 받기
3. **get_structure** → 셀별 콘텐츠 사전 파악 + TOC 확인
4. **get_overview** → 레이아웃 시각적 확인 (선택적 — get_structure만으로 충분하면 생략 가능)
5. **get_tile** / **get_tile_as_pdf** / **get_tile_text** → 필요한 셀만 선택 분석

## 핵심 원칙

- 문서 전체를 한 번에 처리하는 것은 메모리·토큰 한계로 불가능합니다.
- get_structure로 콘텐츠가 있는 셀을 먼저 파악한 뒤, 필요한 셀만 선택적으로 가져오세요.
- grid_rows, grid_cols는 모든 호출에서 동일한 값을 사용해야 셀 번호가 일치합니다.
""",
)


def _validate_grid(grid_rows: int, grid_cols: int) -> None:
    if grid_rows < 1 or grid_cols < 1:
        raise ValueError(f"grid_rows와 grid_cols는 1 이상이어야 합니다. (받은 값: rows={grid_rows}, cols={grid_cols})")


def _validate_cell_idx(cell_idx: int, grid_rows: int, grid_cols: int) -> None:
    max_idx = grid_rows * grid_cols - 1
    if cell_idx < 0 or cell_idx > max_idx:
        raise ValueError(
            f"cell_idx={cell_idx}는 유효 범위(0~{max_idx})를 벗어났습니다. "
            f"grid_rows={grid_rows}, grid_cols={grid_cols} 기준입니다."
        )


def _validate_dpi(dpi: int) -> None:
    if dpi < 1 or dpi > 600:
        raise ValueError(f"dpi는 1~600 범위여야 합니다. (받은 값: {dpi})")


@mcp.tool()
@track_call("get_page_count")
def get_page_count(pdf_path: str) -> str:
    """
    PDF의 페이지 수와 각 페이지 크기를 반환합니다.

    다중 페이지 문서 분석 시 가장 먼저 호출하여
    전체 페이지 수와 각 페이지 크기(pt 단위)를 확인하세요.

    Args:
        pdf_path: PDF 파일 절대 경로
    """
    return json.dumps(_get_page_count(pdf_path), ensure_ascii=False)


@mcp.tool()
@track_call("suggest_grid")
def suggest_grid(
    pdf_path: str,
    page_idx: int = 0,
    target_tile_pt: int = 1500,
) -> str:
    """
    페이지 크기와 텍스트 밀도를 분석하여 적절한 그리드 크기를 추천합니다.

    get_page_count 다음, get_overview 이전에 호출하면 최적의 grid_rows/grid_cols를
    결정하는 데 도움이 됩니다. 특히 문서가 매우 크거나 텍스트가 밀집된 경우 유용합니다.

    반환 필드:
    - page_size_pt: 페이지 크기 (포인트 단위)
    - page_size_mm: 페이지 크기 (밀리미터 단위)
    - text_block_count: 텍스트 블록 수
    - total_chars: 전체 문자 수
    - image_block_count: 이미지 블록 수
    - text_density: 텍스트 밀도 (문자 수 / 페이지 면적)
    - suggested_grid: 추천 그리드 크기 {"rows": ..., "cols": ...}
    - suggested_tile_size_pt: 추천 그리드 사용 시 타일 크기 (포인트 단위)

    Args:
        pdf_path: PDF 파일 절대 경로
        page_idx: 분석할 페이지 번호 (0부터 시작, 기본 0)
        target_tile_pt: 목표 타일 크기 포인트 (기본 1500)
    """
    return json.dumps(_analyze_page(pdf_path, page_idx=page_idx, target_tile_pt=target_tile_pt), ensure_ascii=False)


@mcp.tool()
@track_call("get_overview")
def get_overview(
    pdf_path: str,
    grid_rows: int = 8,
    grid_cols: int = 8,
    max_px: int = 2048,
    page_idx: int = 0,
) -> Image:
    """
    문서 전체를 저해상도 한 장의 이미지로 렌더링하고,
    grid_rows × grid_cols 개의 셀로 나눈 번호 격자를 오버레이하여 반환합니다.
    (기본값 8×8 = 64개 셀, 번호는 0부터 시작)

    반환된 이미지로 레이아웃을 파악한 뒤, 분석이 필요한 영역의 셀 번호를 확인하고
    get_tile 또는 get_tile_as_pdf를 호출하여 해당 부분만 정밀하게 읽으세요.

    이 도구는 선택적입니다. get_structure만으로 콘텐츠 있는 셀을 파악했다면 생략할 수 있습니다.

    Args:
        pdf_path: PDF 파일 절대 경로
        grid_rows: 그리드 행 수 (기본 8, 문서가 세로로 길면 늘리세요)
        grid_cols: 그리드 열 수 (기본 8, 문서가 가로로 길면 늘리세요)
        max_px: overview 이미지의 최대 픽셀 크기 (기본 2048)
        page_idx: 페이지 번호 (0부터 시작, 기본 0)
    """
    _validate_grid(grid_rows, grid_cols)
    overview_bytes = render_overview(pdf_path, max_px=max_px, page_idx=page_idx)
    overlay_bytes, _ = add_grid_overlay(overview_bytes, grid_rows, grid_cols)
    return Image(data=overlay_bytes, format="jpeg")


@mcp.tool()
@track_call("get_tile")
def get_tile(
    pdf_path: str,
    cell_idx: int,
    grid_rows: int = 8,
    grid_cols: int = 8,
    dpi: int = 72,
    page_idx: int = 0,
    overlap: float = 0.0,
) -> Image:
    """
    get_structure 또는 get_overview에서 확인한 특정 셀을 고해상도 이미지로 추출하여 반환합니다.
    문서 전체가 아닌 필요한 셀만 렌더링하므로 메모리를 효율적으로 사용합니다.

    주의: grid_rows, grid_cols는 get_overview 호출 시 사용한 값과 반드시 동일해야 합니다.

    Args:
        pdf_path: PDF 파일 절대 경로
        cell_idx: 추출할 셀 번호 (get_overview 이미지에 표시된 번호)
        grid_rows: 그리드 행 수 (get_overview와 동일하게)
        grid_cols: 그리드 열 수 (get_overview와 동일하게)
        dpi: 렌더링 해상도 (기본 72 DPI — 이 문서는 원본 자체가 초고해상도이므로
             72 DPI도 충분히 선명합니다. 더 높은 해상도가 필요하면 높이세요.)
        page_idx: 페이지 번호 (0부터 시작, 기본 0)
        overlap: 타일 경계 확장 비율 (기본 0.0 — 0.1이면 상하좌우 각 10% 확장,
                 경계에 걸친 텍스트·도형을 양쪽 타일에서 볼 수 있습니다. 권장 범위: 0.0~0.5 미만)
    """
    _validate_grid(grid_rows, grid_cols)
    _validate_cell_idx(cell_idx, grid_rows, grid_cols)
    _validate_dpi(dpi)
    tile_bytes = render_tile(pdf_path, cell_idx, grid_rows, grid_cols, dpi=dpi, page_idx=page_idx, overlap=overlap)
    return Image(data=tile_bytes, format="jpeg")


@mcp.tool()
@track_call("get_tile_as_pdf")
def get_tile_as_pdf(
    pdf_path: str,
    cell_idx: int,
    grid_rows: int = 8,
    grid_cols: int = 8,
    page_idx: int = 0,
    overlap: float = 0.0,
) -> str:
    """
    get_structure 또는 get_overview에서 확인한 특정 셀을 PDF 파일로 추출하고 저장 경로를 반환합니다.
    Ghostscript로 해당 영역 밖의 이미지·폰트를 제거하여 크기를 최소화합니다.

    원본이 벡터 PDF(CAD 도면, 기술 문서 등)일 때 이미지 변환 없이
    텍스트·도형의 품질을 완벽하게 보존할 수 있습니다.
    반환된 경로의 파일을 다른 PDF 처리 도구로 읽거나 전달하세요.

    중요: 반환된 임시 파일은 사용 후 직접 삭제해야 합니다 (os.remove(path)).
    Ghostscript 오류 시 get_tile 도구로 대체할 수 있습니다.

    주의: grid_rows, grid_cols는 get_overview 호출 시 사용한 값과 반드시 동일해야 합니다.
    시스템에 Ghostscript가 설치되어 있어야 합니다 (brew install ghostscript).

    Args:
        pdf_path: PDF 파일 절대 경로
        cell_idx: 추출할 셀 번호 (get_overview 이미지에 표시된 번호)
        grid_rows: 그리드 행 수 (get_overview와 동일하게)
        grid_cols: 그리드 열 수 (get_overview와 동일하게)
        page_idx: 페이지 번호 (0부터 시작, 기본 0)
        overlap: 타일 경계 확장 비율 (기본 0.0 — 0.1이면 상하좌우 각 10% 확장,
                 경계에 걸친 텍스트·도형을 양쪽 타일에서 볼 수 있습니다. 권장 범위: 0.0~0.5 미만)
    """
    _validate_grid(grid_rows, grid_cols)
    _validate_cell_idx(cell_idx, grid_rows, grid_cols)
    pdf_bytes = render_tile_as_pdf(pdf_path, cell_idx, grid_rows, grid_cols, page_idx=page_idx, overlap=overlap)
    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f"_cell{cell_idx}.pdf",
        dir=tempfile.gettempdir(),
    )
    tmp.write(pdf_bytes)
    tmp.close()
    return tmp.name


@mcp.tool()
@track_call("get_tile_text")
def get_tile_text(
    pdf_path: str,
    cell_idx: int,
    grid_rows: int = 8,
    grid_cols: int = 8,
    page_idx: int = 0,
    overlap: float = 0.0,
    format: str = "text",
) -> str:
    """
    특정 셀 영역의 벡터 텍스트 데이터를 JSON 형식으로 반환합니다.

    get_tile(이미지)과 함께 사용하면 LLM이 이미지와 텍스트를 교차 검증하여
    텍스트 인식 정확도를 높일 수 있습니다.
    동일한 cell_idx와 grid 설정을 사용하면 get_tile과 좌표가 일치합니다.

    반환값 구조:
    - cell_idx: 요청한 셀 번호
    - clip_rect: 추출 영역 [x0, y0, x1, y1] (포인트 단위)

    format 옵션:
    - "text" (기본): 평문 텍스트만 반환 (구조 없음). 내용만 빠르게 읽을 때 사용.
    - "compact": 블록·라인·span 구조 유지, bbox + text만 포함. 위치 기반 분석에 적합.

    주의: grid_rows, grid_cols는 get_overview 호출 시 사용한 값과 반드시 동일해야 합니다.

    Args:
        pdf_path: PDF 파일 절대 경로
        cell_idx: 추출할 셀 번호 (get_overview 이미지에 표시된 번호)
        grid_rows: 그리드 행 수 (get_overview와 동일하게)
        grid_cols: 그리드 열 수 (get_overview와 동일하게)
        page_idx: 페이지 번호 (0부터 시작, 기본 0)
        overlap: 타일 경계 확장 비율 (기본 0.0 — get_tile과 동일한 값을 사용하세요)
        format: 출력 형식 — "compact"(기본) 또는 "text"
    """
    _validate_grid(grid_rows, grid_cols)
    _validate_cell_idx(cell_idx, grid_rows, grid_cols)
    result = extract_tile_text(pdf_path, cell_idx, grid_rows, grid_cols, page_idx=page_idx, overlap=overlap, format=format)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
@track_call("get_structure")
def get_structure(
    pdf_path: str,
    grid_rows: int = 8,
    grid_cols: int = 8,
    page_idx: int = 0,
    overlap: float = 0.0,
    include_empty: bool = False,
) -> str:
    """
    셀별 텍스트/이미지 유무와 PDF 목차를 JSON으로 반환합니다.

    get_overview 이전 또는 이후에 호출하면 어느 셀을 get_tile로 분석할지
    사전에 파악할 수 있어 불필요한 호출을 줄일 수 있습니다.
    grid_rows, grid_cols는 get_overview와 동일한 값을 사용해야 셀 번호가 일치합니다.

    반환값 구조:
    - cells: 셀 번호 → {has_text, has_image, text_preview} 매핑
      - has_text: 텍스트 블록 존재 여부
      - has_image: 이미지 블록 존재 여부
      - text_preview: 셀 내 첫 텍스트 미리보기 (최대 80자)
    - toc: PDF 목차 [[level, title, page], ...] (없으면 빈 리스트)
      ※ toc의 page 값은 1-indexed입니다. page_idx 파라미터(0-indexed)로 사용하려면
        반드시 1을 빼야 합니다. 예: toc_page=3 → page_idx=2

    활용 방법:
    - has_text=true → get_tile_text로 빠른 텍스트 추출 우선
    - has_image=true → get_tile(이미지) 또는 get_tile_as_pdf로 시각 분석
    - toc → 전체 문서 구조를 파악하여 관심 페이지로 바로 이동

    Args:
        pdf_path: PDF 파일 절대 경로
        grid_rows: 그리드 행 수 (get_overview와 동일하게, 기본 8)
        grid_cols: 그리드 열 수 (get_overview와 동일하게, 기본 8)
        page_idx: 페이지 번호 (0부터 시작, 기본 0)
        overlap: 타일 경계 확장 비율 (기본 0.0 — get_tile/get_tile_text와 동일한 값을 사용하세요.
                 overlap 사용 시 경계 근처 콘텐츠도 정확히 감지됩니다.)
        include_empty: False(기본)이면 비어 있는 셀을 결과에서 제외합니다.
                       빈 셀까지 모두 보려면 True로 설정하세요.
    """
    _validate_grid(grid_rows, grid_cols)
    result = _get_page_structure(pdf_path, grid_rows, grid_cols, page_idx, overlap=overlap, include_empty=include_empty)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
@track_call("find_cells")
def find_cells(
    pdf_path: str,
    query: str,
    grid_rows: int = 8,
    grid_cols: int = 8,
    page_idx: int = 0,
    overlap: float = 0.0,
    case_sensitive: bool = False,
) -> str:
    """
    특정 문자열이 포함된 셀 번호 목록을 반환합니다.

    LLM이 키워드가 어느 셀에 있는지 한 번에 찾을 수 있어,
    모든 셀을 get_tile_text로 개별 호출하는 비효율을 줄입니다.
    반환된 matched_cells 번호로 get_tile_text를 호출하여 상세 내용을 확인하세요.

    반환값 구조:
    - query: 검색한 문자열
    - matched_cells: 해당 문자열이 포함된 셀 번호 목록

    Args:
        pdf_path: PDF 파일 절대 경로
        query: 검색할 문자열
        grid_rows: 그리드 행 수 (get_overview와 동일하게, 기본 8)
        grid_cols: 그리드 열 수 (get_overview와 동일하게, 기본 8)
        page_idx: 페이지 번호 (0부터 시작, 기본 0)
        overlap: 타일 경계 확장 비율 (기본 0.0)
        case_sensitive: 대소문자 구분 여부 (기본 False — 구분 안 함)
    """
    _validate_grid(grid_rows, grid_cols)
    result = _search_cells(pdf_path, query, grid_rows, grid_cols, page_idx=page_idx, overlap=overlap, case_sensitive=case_sensitive)
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    MonitorServer().start()
    mcp.run()
