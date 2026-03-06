import json

import fitz
from mcp.server.fastmcp import FastMCP, Image

from monitor import MonitorServer, track_call
from pdf_tiler import _extract_text_clip
from pdf_tiler import detect_content_regions as _detect_content_regions
from pdf_tiler import get_page_count as _get_page_count
from pdf_tiler import render_region as _render_region

mcp = FastMCP(
    "pdfllm",
    instructions="""이 서버는 LLM이 직접 읽기 어려운 대용량 PDF(수십~수백 MB, 수억 픽셀 규모)를
효율적으로 분석하기 위한 도구를 제공합니다.

## 워크플로우

1. **get_page_count** → 페이지 수·크기 파악
2. **detect_layout** → 콘텐츠 경계에 맞는 region 목록 확인
3. **get_region** → 각 region 이미지·텍스트 추출

콘텐츠 경계에 맞게 region을 자동 분리하므로 하나의 콘텐츠가 잘리는 문제가 없습니다.
문서 전체를 한 번에 처리하는 것은 메모리·토큰 한계로 불가능합니다.
""",
)


@mcp.tool()
@track_call("get_page_count")
def get_page_count(pdf_path: str) -> str:
    """
    PDF의 페이지 수와 각 페이지 크기를 반환합니다.

    다중 페이지 문서 분석 시 가장 먼저 호출하여
    전체 페이지 수와 각 페이지 크기(pt 단위)를 확인하세요.

    반환 예시: {"page_count": 3, "pages": [{"width": 595.0, "height": 842.0}, ...]}

    Args:
        pdf_path: PDF 파일 절대 경로
    """
    return json.dumps(_get_page_count(pdf_path), ensure_ascii=False)


@mcp.tool()
@track_call("detect_layout")
def detect_layout(
    pdf_path: str,
    page_idx: int = 0,
    min_gap_pt: float = 30.0,
    padding_pt: float = 6.0,
) -> str:
    """
    PyMuPDF 기반 whitespace 분석으로 콘텐츠 region 목록을 반환합니다.

    고정 그리드 타일링과 달리 콘텐츠 경계에 맞게 region을 자동 분리하므로,
    하나의 콘텐츠가 여러 타일에 걸쳐 잘리는 문제가 없습니다.

    반환: {"page_idx", "page_size_pt": [w, h], "regions": [{"idx", "bbox_pt": [x0,y0,x1,y1]}, ...]}

    반환된 regions[i]["bbox_pt"]를 get_region에 전달하여 각 region을 이미지·텍스트로 추출하세요.

    Args:
        pdf_path: PDF 파일 절대 경로
        page_idx: 페이지 번호 (0부터 시작, 기본 0)
        min_gap_pt: 콘텐츠 분리 기준 최소 여백 크기(pt). 기본 30pt.
                    넓은 섹션 단위로 분리하려면 100~200, 세밀하게는 20~30.
        padding_pt: 각 region bbox 확장 패딩(pt). 기본 6pt.
    """
    regions = _detect_content_regions(pdf_path, page_idx, min_gap_pt, padding_pt)
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    result = {
        "page_idx": page_idx,
        "page_size_pt": [page.rect.width, page.rect.height],
        "regions": regions,
    }
    doc.close()
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
@track_call("get_region")
def get_region(
    pdf_path: str,
    page_idx: int,
    bbox_pt: list[float],
    dpi: int = 72,
    output: str = "image",
    padding_pt: float = 6.0,
):
    """
    detect_layout의 bbox_pt를 받아 이미지 또는 텍스트로 반환합니다.

    Args:
        pdf_path: PDF 파일 절대 경로
        page_idx: 페이지 번호 (0부터 시작)
        bbox_pt: [x0, y0, x1, y1] (포인트 단위) — detect_layout regions[i]["bbox_pt"]
        dpi: 렌더링 해상도 (output="image"일 때만 사용, 기본 72)
        output: "image"(기본) → JPEG Image 반환 / "text" → {"bbox_pt": [...], "text": "..."} JSON 반환
        padding_pt: bbox 주변 추가 패딩(pt). 기본 6pt.
    """
    if output == "text":
        clip = fitz.Rect(bbox_pt[0], bbox_pt[1], bbox_pt[2], bbox_pt[3])
        text = _extract_text_clip(pdf_path, clip, page_idx)
        return json.dumps({"bbox_pt": bbox_pt, "text": text}, ensure_ascii=False)
    img_bytes = _render_region(
        pdf_path, page_idx, bbox_pt, dpi=dpi, padding_pt=padding_pt
    )
    return Image(data=img_bytes, format="jpeg")


def main():
    MonitorServer().start()
    mcp.run()


if __name__ == "__main__":
    main()
