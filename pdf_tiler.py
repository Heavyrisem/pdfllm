import shutil
import subprocess
import tempfile
from pathlib import Path

import fitz  # PyMuPDF


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
    dpi: int = 300,
    page_idx: int = 0,
) -> bytes:
    """
    그리드 셀 인덱스에 해당하는 영역을 고해상도로 렌더링하여 JPEG bytes 반환.
    메모리 누수 방지를 위해 처리 후 즉시 해제.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    rect = page.rect

    row, col = divmod(cell_idx, cols)
    tile_w = rect.width / cols
    tile_h = rect.height / rows
    clip = fitz.Rect(
        col * tile_w,
        row * tile_h,
        (col + 1) * tile_w,
        (row + 1) * tile_h,
    )

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

    row, col = divmod(cell_idx, cols)
    tile_w = rect.width / cols
    tile_h = rect.height / rows
    clip = fitz.Rect(
        col * tile_w,
        row * tile_h,
        (col + 1) * tile_w,
        (row + 1) * tile_h,
    )

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
        )

        return out_path.read_bytes()


def get_page_size(pdf_path: str, page_idx: int = 0) -> tuple[float, float]:
    """PDF 페이지의 가로×세로 크기를 포인트 단위로 반환."""
    doc = fitz.open(pdf_path)
    rect = doc[page_idx].rect
    doc.close()
    return rect.width, rect.height
