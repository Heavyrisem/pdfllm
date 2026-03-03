# pdfllm

LLM이 대용량 PDF(수십~수백 MB)를 효율적으로 분석할 수 있도록 타일 단위로 분할하는 FastMCP 기반 MCP 서버.

---

## Claude Code에 Local MCP로 등록하기 (uvx)

`uvx`를 사용하면 별도 설치 없이 로컬 패키지를 바로 MCP 서버로 실행할 수 있습니다.

### 방법 1: `claude mcp add` 명령어 (권장)

```bash
claude mcp add pdfllm -s local -- uvx --from /절대경로/pdfllm pdfllm
```

예시:

```bash
claude mcp add pdfllm -s local -- uvx --from /Users/yourname/Desktop/dev/pdfllm pdfllm
```

> **`-s` 옵션 설명:**
> - `-s local` — 현재 사용자의 모든 프로젝트에서 사용 가능 (`~/.claude.json`에 저장)
> - `-s project` — 현재 프로젝트에서만 사용 가능 (`.claude/settings.json`에 저장)

### 방법 2: `~/.claude.json` 직접 편집

`~/.claude.json` 파일의 `mcpServers` 항목에 아래 내용을 추가합니다:

```json
{
  "mcpServers": {
    "pdfllm": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "/절대경로/pdfllm", "pdfllm"]
    }
  }
}
```

---

## 등록 확인

```bash
# 등록된 MCP 서버 목록 확인
claude mcp list

# 특정 서버 상세 확인
claude mcp get pdfllm
```

Claude Code를 재시작하면 MCP 서버가 자동으로 연결됩니다.

---

## 제공 도구

| 도구 | 설명 |
|------|------|
| `get_page_count` | 페이지 수·크기 반환 |
| `suggest_grid` | 텍스트 밀도 기반 최적 grid 추천 |
| `get_overview` | 전체 페이지 저해상도 + 번호 격자 오버레이 |
| `get_structure` | 셀별 텍스트/이미지 유무 + TOC 반환 |
| `get_tile` | 특정 셀 고해상도 이미지 반환 |
| `get_tile_as_pdf` | 특정 셀 PDF 파일로 저장 (Ghostscript 설치 시 사용 가능: `brew install ghostscript`) |
| `get_tile_text` | 특정 셀 벡터 텍스트 추출 |
| `find_cells` | 키워드 포함 셀 번호 목록 반환 |

## 권장 워크플로우

```
get_page_count → suggest_grid → get_structure → (get_overview) → get_tile / get_tile_text
```

1. `get_page_count` - 페이지 수·크기 파악
2. `suggest_grid` - 적절한 grid_rows/grid_cols 추천 받기
3. `get_structure` - 셀별 콘텐츠 사전 파악 + TOC 확인
4. `get_overview` - 레이아웃 시각적 확인 (선택적)
5. `get_tile` / `get_tile_text` - 필요한 셀만 선택 분석
