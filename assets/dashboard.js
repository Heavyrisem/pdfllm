const App = (() => {
  const IMAGE_TOOLS = new Set(["get_region"]);
  const cachedImages = {};
  let selectedInstance = null;
  let selfInstanceId = null;
  let knownTools = [];

  function fmtUptime(s) {
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    if (h) return h + 'h ' + String(m).padStart(2, '0') + 'm ' + String(sec).padStart(2, '0') + 's';
    return String(m).padStart(2, '0') + 'm ' + String(sec).padStart(2, '0') + 's';
  }

  function ago(ts) {
    const d = Math.floor(Date.now() / 1000 - ts);
    if (d < 60) return d + '초 전';
    if (d < 3600) return Math.floor(d / 60) + '분 전';
    // 1시간 이상: 정확한 시각 표시
    const dt = new Date(ts * 1000);
    return dt.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' });
  }

  function escHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function renderMinimap(bbox_pt, page_size_pt) {
    const [x0, y0, x1, y1] = bbox_pt;
    const [pw, ph] = page_size_pt;
    const W = 48, H = Math.max(4, Math.round(48 * ph / pw));
    const rx = (x0 / pw * W).toFixed(1);
    const ry = (y0 / ph * H).toFixed(1);
    const rw = ((x1 - x0) / pw * W).toFixed(1);
    const rh = ((y1 - y0) / ph * H).toFixed(1);
    return `<svg class="call-minimap" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
      <rect width="${W}" height="${H}" fill="#161b22" rx="2" stroke="#30363d" stroke-width="1"/>
      <rect x="${rx}" y="${ry}" width="${rw}" height="${rh}"
            fill="rgba(88,166,255,0.3)" stroke="#58a6ff" stroke-width="1"/>
    </svg>`;
  }

  async function fetchImage(callId) {
    if (cachedImages[callId]) return cachedImages[callId];
    try {
      const r = await fetch('/api/image/' + callId);
      if (!r.ok) return null;
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      cachedImages[callId] = url;
      return url;
    } catch { return null; }
  }

  async function render(data) {
    // Uptime
    document.getElementById('uptime').textContent = fmtUptime(data.uptime_s);

    // 동적 도구 목록 (서버에서 수신, 없으면 이전 값 유지)
    const allTools = (data.all_tools && data.all_tools.length) ? data.all_tools : knownTools;
    if (allTools.length) knownTools = allTools;

    // Tool stats
    const stats = data.tool_stats || {};
    const statsHtml = knownTools.map(t => {
      const s = stats[t];
      if (!s) return `<div class="tool-card"><div class="tool-name">${escHtml(t)}</div><div class="tool-meta" style="color:#6e7681">미사용</div></div>`;
      const avg = s.avg_ms != null ? s.avg_ms + 'ms' : '-';
      const minMs = s.min_ms != null ? s.min_ms + 'ms' : '-';
      const maxMs = s.max_ms != null ? s.max_ms + 'ms' : '-';
      const errRate = s.total ? Math.round(s.error / s.total * 100) : 0;
      return `<div class="tool-card">
        <div class="tool-name">${escHtml(t)}</div>
        <div class="tool-meta">
          호출: ${s.total} &nbsp;
          <span class="s-ok">✓${s.success}</span>
          ${s.error ? `<span class="s-err"> ✗${s.error} (${errRate}%)</span>` : ''}
          <br>min: ${minMs} / avg: ${avg} / max: ${maxMs}
        </div>
      </div>`;
    }).join('');
    document.getElementById('tool-stats').innerHTML = statsHtml;

    // Active calls
    const active = data.active_calls || [];
    if (active.length === 0) {
      document.getElementById('active-calls').innerHTML = '<span class="empty">실행 중인 도구 없음</span>';
    } else {
      const now = Date.now() / 1000;
      document.getElementById('active-calls').innerHTML = active.map(c => {
        const elapsed = (now - c.start_time).toFixed(1) + 's';
        return `<div class="active-row">
          <div class="spinner"></div>
          <span class="active-tool">${escHtml(c.tool)}</span>
          <span class="active-params">${escHtml(c.params)}</span>
          <span class="active-elapsed">${elapsed}</span>
        </div>`;
      }).join('');
    }

    // Recent calls
    const recent = data.recent_calls || [];
    if (recent.length === 0) {
      document.getElementById('recent-calls').innerHTML = '<span class="empty">호출 기록 없음</span>';
      return;
    }

    // 이미지 도구 결과 미리 fetch
    const imagePromises = recent
      .filter(c => c.has_image && !cachedImages[c.id])
      .map(c => fetchImage(c.id));
    await Promise.all(imagePromises);

    const cards = recent.map(c => {
      const icon = c.status === 'success' ? '✓' : c.status === 'error' ? '✗' : '⏳';
      const dur = c.duration_ms != null ? c.duration_ms + 'ms' : '-';
      const errHtml = c.error ? `<div class="call-error">오류: ${escHtml(c.error)}</div>` : '';
      const imgUrl = cachedImages[c.id];
      const imgHtml = imgUrl ? `<img class="call-img" src="${imgUrl}" loading="lazy">` : '';
      const minimapHtml = (c.tool === 'get_region' && c.bbox_pt && c.page_size_pt)
        ? renderMinimap(c.bbox_pt, c.page_size_pt) : '';
      return `<div class="call-card${c.status === 'error' ? ' err' : ''}">
        <div class="call-header">
          ${minimapHtml}
          <span class="call-icon ${c.status === 'success' ? 's-ok' : c.status === 'error' ? 's-err' : ''}">${icon}</span>
          <span class="call-tool">${escHtml(c.tool)}</span>
          <span class="call-dur">${dur}</span>
          <span class="call-ago">${ago(c.start_time)}</span>
        </div>
        <div class="call-params">${escHtml(c.params)}</div>
        ${errHtml}
        ${imgHtml}
      </div>`;
    }).join('');
    document.getElementById('recent-calls').innerHTML = cards;
  }

  function renderTabs(data) {
    const instances = data.instances || {};
    const now = Date.now() / 1000;
    const html = Object.entries(instances).map(([iid, s]) => {
      const isMaster = iid === data.self_instance_id;
      const lastSeen = s.last_seen || now;
      const offline = !isMaster && (now - lastSeen >= 5);
      const label = isMaster ? `${iid} (master)` : iid;
      const isActive = selectedInstance === iid;
      const classes = ['tab', isActive ? 'active' : '', offline ? 'offline' : ''].filter(Boolean).join(' ');
      return `<div class="${classes}" onclick="App.selectInstance('${escHtml(iid)}')">${escHtml(label)}</div>`;
    }).join('');
    document.getElementById('instance-tabs').innerHTML = html;
  }

  function selectInstance(iid) {
    selectedInstance = iid;
    document.querySelectorAll('.tab').forEach(el => {
      el.classList.toggle('active', el.textContent.startsWith(iid));
    });
  }

  async function poll() {
    try {
      const r = await fetch('/api/status');
      if (r.ok) {
        const data = await r.json();
        selfInstanceId = data.self_instance_id;
        if (!selectedInstance || !data.instances[selectedInstance]) {
          selectedInstance = selfInstanceId;
        }
        renderTabs(data);
        const instanceData = data.instances[selectedInstance];
        if (instanceData) await render(instanceData);
      }
    } catch {}
    setTimeout(poll, 1000);
  }

  poll();

  return { selectInstance };
})();
