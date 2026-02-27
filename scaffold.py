import io
from PIL import Image, ImageDraw, ImageFont


# 플랫폼별 후보 폰트 경로 (앞쪽부터 순서대로 시도)
_FONT_CANDIDATES = [
    # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Arial.ttf",
    # Linux (Debian/Ubuntu)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    # Windows
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """플랫폼에 맞는 트루타입 폰트를 로드한다. 없으면 기본 폰트 사용."""
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def add_grid_overlay(
    img_bytes: bytes,
    rows: int,
    cols: int,
) -> tuple[bytes, dict[int, tuple[int, int, int, int]]]:
    """
    이미지에 번호 격자 오버레이를 추가합니다.

    Returns:
        (오버레이 이미지 JPEG bytes, {셀번호: (x1, y1, x2, y2) 픽셀 좌표} 딕셔너리)
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    w, h = img.size

    # 반투명 오버레이 레이어
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 이미지 크기에 비례한 폰트 크기
    font_size = max(10, min(w, h) // (max(rows, cols) * 2))
    font = _load_font(font_size)

    cell_map: dict[int, tuple[int, int, int, int]] = {}

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            x1 = c * w // cols
            y1 = r * h // rows
            x2 = (c + 1) * w // cols
            y2 = (r + 1) * h // rows

            # 반투명 선 (alpha=120)
            draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=(255, 50, 50, 120), width=2)
            # 반투명 텍스트 (alpha=160)
            draw.text((x1 + 4, y1 + 4), str(idx), fill=(255, 50, 50, 160), font=font)

            cell_map[idx] = (x1, y1, x2, y2)

    img = Image.alpha_composite(img, overlay).convert("RGB")
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=85)
    return output.getvalue(), cell_map


def save_debug_image(img_bytes: bytes, path: str) -> None:
    """디버그용: 오버레이 이미지를 파일로 저장."""
    with open(path, "wb") as f:
        f.write(img_bytes)
