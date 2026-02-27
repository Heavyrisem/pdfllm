# pdfllm - PDF Tiling MCP Server

FastMCP 기반 PDF 타일링 MCP 서버. LLM이 대용량 PDF를 효율적으로 분석할 수 있도록 타일 단위로 분할하는 도구 모음.

## Commands

```bash
# 의존성 설치
pip install -r requirements.txt
brew install ghostscript  # get_tile_as_pdf에 필요

# 서버 실행
.venv/bin/python server.py
```

## Architecture

3개 파일로 구성:

- **`server.py`** - MCP 도구 정의 (FastMCP 엔드포인트)
- **`pdf_tiler.py`** - 핵심 로직 (타일 좌표 계산, 렌더링, 텍스트 추출)
- **`scaffold.py`** - 그리드 오버레이 (get_overview용 번호 격자 생성)

### 8개 MCP 도구

| 도구 | 설명 |
|------|------|
| `get_page_count` | 페이지 수·크기 반환 |
| `suggest_grid` | 텍스트 밀도 기반 최적 grid 추천 |
| `get_overview` | 전체 페이지 저해상도 + 번호 격자 오버레이 |
| `get_tile` | 특정 셀 고해상도 이미지 반환 |
| `get_tile_as_pdf` | 특정 셀 PDF 파일로 저장 (Ghostscript 필요) |
| `get_tile_text` | 특정 셀 벡터 텍스트 추출 |
| `get_structure` | 셀별 텍스트/이미지 유무 + TOC 반환 |
| `find_cells` | 키워드 포함 셀 번호 목록 반환 |

### 권장 워크플로우

```
get_page_count → suggest_grid → get_structure → (get_overview) → get_tile / get_tile_text
```

## 중요 구현 세부사항

- **`grid_rows/cols` 일관성**: 모든 도구 호출에서 동일한 값 사용해야 셀 번호가 일치함
- **TOC page 인덱싱**: TOC의 `page`는 1-indexed, `page_idx` 파라미터는 0-indexed (1 차이)
- **`overlap` 파라미터**: 0.0~0.5 범위, 경계 텍스트 처리용 타일 확장 비율
- **`get_tile_text` 포맷**: `"text"`(기본, 빠름) / `"compact"`(bbox 위치 정보 포함)
- **좌표 변환**: `_calc_cell_clip()` 함수가 모든 도구의 셀 → 좌표 변환에 사용됨
- **메모리 관리**: `del pix` + `fitz.TOOLS.store_shrink(100)` 패턴으로 메모리 해제
