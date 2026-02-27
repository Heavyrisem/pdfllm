import json
import tempfile
import os

from mcp.server.fastmcp import FastMCP, Image

from pdf_tiler import render_overview, render_tile, render_tile_as_pdf, get_page_count as _get_page_count, extract_tile_text
from scaffold import add_grid_overlay

mcp = FastMCP(
    "pdfllm",
    instructions="""이 서버는 LLM이 직접 읽기 어려운 대용량 PDF(수십~수백 MB, 수억 픽셀 규모)를
효율적으로 분석하기 위한 타일링 도구를 제공합니다.

## 권장 워크플로우

1. **get_page_count** 호출 → 총 페이지 수와 각 페이지 크기 파악 (다중 페이지 문서일 경우)
2. **get_overview** 호출 → 문서 전체를 저해상도로 확인하고 번호가 매겨진 그리드로 레이아웃 파악
3. 분석이 필요한 셀 번호 식별
4. **get_tile** 또는 **get_tile_as_pdf** 호출 → 해당 셀만 고해상도로 추출하여 정밀 분석

## 핵심 원칙

- 문서 전체를 한 번에 처리하는 것은 메모리·토큰 한계로 불가능합니다.
- get_overview로 관련 영역을 먼저 특정한 뒤, 필요한 셀만 선택적으로 가져오세요.
- grid_rows, grid_cols는 모든 호출에서 동일한 값을 사용해야 셀 번호가 일치합니다.
""",
)


@mcp.tool()
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
def get_overview(
    pdf_path: str,
    grid_rows: int = 8,
    grid_cols: int = 8,
    max_px: int = 2048,
    page_idx: int = 0,
) -> Image:
    """
    대용량 PDF 분석의 첫 번째 단계입니다.

    문서 전체를 저해상도 한 장의 이미지로 렌더링하고,
    grid_rows × grid_cols 개의 셀로 나눈 번호 격자를 오버레이하여 반환합니다.
    (기본값 8×8 = 64개 셀, 번호는 0부터 시작)

    반환된 이미지로 레이아웃을 파악한 뒤, 분석이 필요한 영역의 셀 번호를 확인하고
    get_tile 또는 get_tile_as_pdf를 호출하여 해당 부분만 정밀하게 읽으세요.

    Args:
        pdf_path: PDF 파일 절대 경로
        grid_rows: 그리드 행 수 (기본 8, 문서가 세로로 길면 늘리세요)
        grid_cols: 그리드 열 수 (기본 8, 문서가 가로로 길면 늘리세요)
        max_px: overview 이미지의 최대 픽셀 크기 (기본 2048)
        page_idx: 페이지 번호 (0부터 시작, 기본 0)
    """
    overview_bytes = render_overview(pdf_path, max_px=max_px, page_idx=page_idx)
    overlay_bytes, _ = add_grid_overlay(overview_bytes, grid_rows, grid_cols)
    return Image(data=overlay_bytes, format="jpeg")


@mcp.tool()
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
    대용량 PDF 분석의 두 번째 단계입니다.

    get_overview에서 확인한 특정 셀을 고해상도 이미지로 추출하여 반환합니다.
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
    tile_bytes = render_tile(pdf_path, cell_idx, grid_rows, grid_cols, dpi=dpi, page_idx=page_idx, overlap=overlap)
    return Image(data=tile_bytes, format="jpeg")


@mcp.tool()
def get_tile_as_pdf(
    pdf_path: str,
    cell_idx: int,
    grid_rows: int = 8,
    grid_cols: int = 8,
    page_idx: int = 0,
    overlap: float = 0.0,
) -> str:
    """
    대용량 PDF 분석의 두 번째 단계 (PDF 출력 버전)입니다.

    get_overview에서 확인한 특정 셀을 PDF 파일로 추출하고 저장 경로를 반환합니다.
    Ghostscript로 해당 영역 밖의 이미지·폰트를 제거하여 크기를 최소화합니다.

    원본이 벡터 PDF(CAD 도면, 기술 문서 등)일 때 이미지 변환 없이
    텍스트·도형의 품질을 완벽하게 보존할 수 있습니다.
    반환된 경로의 파일을 다른 PDF 처리 도구로 읽거나 전달하세요.

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
    pdf_bytes = render_tile_as_pdf(pdf_path, cell_idx, grid_rows, grid_cols, page_idx=page_idx, overlap=overlap)
    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f"_cell{cell_idx}.pdf",
        dir=os.path.dirname(pdf_path),
    )
    tmp.write(pdf_bytes)
    tmp.close()
    return tmp.name


@mcp.tool()
def get_tile_text(
    pdf_path: str,
    cell_idx: int,
    grid_rows: int = 8,
    grid_cols: int = 8,
    page_idx: int = 0,
    overlap: float = 0.0,
) -> str:
    """
    특정 셀 영역의 벡터 텍스트 데이터를 JSON 형식으로 반환합니다.

    get_tile(이미지)과 함께 사용하면 LLM이 이미지와 텍스트를 교차 검증하여
    텍스트 인식 정확도를 높일 수 있습니다.
    동일한 cell_idx와 grid 설정을 사용하면 get_tile과 좌표가 일치합니다.

    반환값 구조:
    - blocks: fitz 텍스트 블록 리스트 (span별 좌표·폰트·텍스트 포함)
    - cell_idx: 요청한 셀 번호
    - clip_rect: 추출 영역 [x0, y0, x1, y1] (포인트 단위)

    주의: grid_rows, grid_cols는 get_overview 호출 시 사용한 값과 반드시 동일해야 합니다.

    Args:
        pdf_path: PDF 파일 절대 경로
        cell_idx: 추출할 셀 번호 (get_overview 이미지에 표시된 번호)
        grid_rows: 그리드 행 수 (get_overview와 동일하게)
        grid_cols: 그리드 열 수 (get_overview와 동일하게)
        page_idx: 페이지 번호 (0부터 시작, 기본 0)
        overlap: 타일 경계 확장 비율 (기본 0.0 — get_tile과 동일한 값을 사용하세요)
    """
    result = extract_tile_text(pdf_path, cell_idx, grid_rows, grid_cols, page_idx=page_idx, overlap=overlap)
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
