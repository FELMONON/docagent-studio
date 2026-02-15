function $(id) { return document.getElementById(id); }

function tabInit() {
  const tabs = document.querySelectorAll(".tab");
  for (const t of tabs) {
    t.addEventListener("click", () => {
      for (const x of tabs) x.classList.remove("active");
      t.classList.add("active");
      const name = t.dataset.tab;
      for (const p of document.querySelectorAll(".panel")) p.classList.remove("active");
      document.querySelector(`#panel-${name}`).classList.add("active");
    });
  }
}

function log(msg) {
  const el = $("workspaceLog");
  const now = new Date().toLocaleTimeString();
  el.textContent = `[${now}] ${msg}\n` + el.textContent;
}

async function apiJson(path, payload) {
  const r = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload || {}),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok || data.ok === false) {
    const err = data.error || `${r.status} ${r.statusText}`;
    throw new Error(err);
  }
  return data;
}

async function apiForm(path, formData) {
  const r = await fetch(path, { method: "POST", body: formData });
  const data = await r.json().catch(() => ({}));
  if (!r.ok || data.ok === false) {
    const err = data.error || `${r.status} ${r.statusText}`;
    throw new Error(err);
  }
  return data;
}

function getCfg() {
  return {
    db_path: $("dbPath").value.trim(),
    embed_model: $("embedModel").value.trim(),
    base_url: $("ollamaBaseUrl").value.trim(),
    model: $("ollamaModel").value.trim(),
    temperature: parseFloat($("temperature").value.trim() || "0.2"),
  };
}

async function refreshStatus() {
  const dot = $("statusDot");
  const txt = $("statusText");
  try {
    const baseUrl = $("ollamaBaseUrl").value.trim();
    const r = await fetch(`/api/health?base_url=${encodeURIComponent(baseUrl)}`);
    const data = await r.json();
    if (!data.ollama_ok) throw new Error(data.error || "Ollama not reachable");
    dot.className = "dot dot-ok";
    txt.textContent = `Ollama OK (${(data.models || []).length} models)`;
  } catch (e) {
    dot.className = "dot dot-bad";
    txt.textContent = `Ollama not ready: ${e.message}`;
  }
}

async function onIngest() {
  const cfg = getCfg();
  const inputDir = $("inputDir").value.trim();
  const notionRoot = $("notionRoot").value.trim();
  const files = $("fileUpload").files;

  const fd = new FormData();
  fd.append("db_path", cfg.db_path);
  fd.append("input_dir", inputDir);
  fd.append("notion_root", notionRoot);
  fd.append("max_chunk_chars", "2500");
  fd.append("overlap", "200");
  for (const f of files) fd.append("files", f);

  log("Ingesting…");
  const t0 = performance.now();
  const data = await apiForm("/api/ingest", fd);
  const dt = ((performance.now() - t0) / 1000).toFixed(2);
  log(`Ingest OK in ${dt}s: docs_seen=${data.documents_seen}, changed=${data.documents_changed}, chunks=${data.chunks_inserted}`);
}

async function onIndex() {
  const cfg = getCfg();
  log("Building index…");
  const t0 = performance.now();
  const data = await apiJson("/api/index", {db_path: cfg.db_path, embed_model: cfg.embed_model, batch_size: 64});
  const dt = ((performance.now() - t0) / 1000).toFixed(2);
  log(`Index OK in ${dt}s: chunks=${data.num_chunks}, dim=${data.embedding_dim}`);
}

async function onGraphBuild() {
  const cfg = getCfg();
  log("Building knowledge graph…");
  const data = await apiJson("/api/graph/build", {db_path: cfg.db_path, clear: true, min_chars: 3, max_per_chunk: 25});
  log(`Graph OK: entities=${data.stats.unique_entities}, edges=${data.stats.unique_edges}, chunks_seen=${data.stats.chunks_seen}`);
}

function renderSources(sources) {
  const el = $("sources");
  if (!sources || sources.length === 0) {
    el.textContent = "";
    return;
  }
  el.textContent = `Sources:\n- ${sources.join("\n- ")}`;
}

async function onAsk() {
  const cfg = getCfg();
  const q = $("question").value.trim();
  const k = parseInt($("topK").value.trim() || "8", 10);
  if (!q) return;

  $("answer").textContent = "Thinking…";
  $("sources").textContent = "";

  const data = await apiJson("/api/ask", {
    db_path: cfg.db_path,
    question: q,
    k,
    model: cfg.model,
    base_url: cfg.base_url,
    temperature: cfg.temperature,
  });
  $("answer").textContent = data.answer || "";
  renderSources(data.sources || []);
}

function hitCard(hit) {
  const el = document.createElement("div");
  el.className = "hit";

  const top = document.createElement("div");
  top.className = "hitTop";
  const src = document.createElement("div");
  src.className = "hitSrc";
  src.textContent = hit.source_ref;
  const score = document.createElement("div");
  score.className = "hitScore";
  score.textContent = `score ${hit.score.toFixed(3)}`;
  top.appendChild(src);
  top.appendChild(score);

  const prev = document.createElement("div");
  prev.className = "hitPrev";
  prev.textContent = hit.preview;

  const btn = document.createElement("button");
  btn.className = "btn hitBtn";
  btn.textContent = "Show chunk";
  btn.addEventListener("click", async () => {
    const cfg = getCfg();
    const url = `/api/chunk?db_path=${encodeURIComponent(cfg.db_path)}&chunk_id=${encodeURIComponent(hit.chunk_id)}`;
    const r = await fetch(url);
    const data = await r.json();
    if (!r.ok || data.ok === false) {
      alert(data.error || "Failed to load chunk");
      return;
    }
    alert(`${data.chunk.source_ref}\n\n${data.chunk.text}`);
  });

  el.appendChild(top);
  el.appendChild(prev);
  el.appendChild(btn);
  return el;
}

async function onSearch() {
  const cfg = getCfg();
  const q = $("searchQuery").value.trim();
  if (!q) return;
  const resEl = $("searchResults");
  resEl.textContent = "Searching…";
  const data = await apiJson("/api/search", {db_path: cfg.db_path, query: q, k: 10});
  resEl.textContent = "";
  for (const h of data.hits || []) resEl.appendChild(hitCard(h));
}

function graphEntityCard(item) {
  const ent = item.entity;
  const el = document.createElement("div");
  el.className = "hit";

  const top = document.createElement("div");
  top.className = "hitTop";
  const src = document.createElement("div");
  src.className = "hitSrc";
  src.textContent = `${ent.name}`;
  const score = document.createElement("div");
  score.className = "hitScore";
  score.textContent = `chunks ${ent.chunk_count} · mentions ${ent.mention_count}`;
  top.appendChild(src);
  top.appendChild(score);

  const prev = document.createElement("div");
  prev.className = "hitPrev";
  const neigh = (item.neighbors || []).slice(0, 6).map(n => `${n.entity.name} (${n.weight})`).join(" · ");
  prev.textContent = neigh ? `Neighbors: ${neigh}` : "No neighbors found.";

  const chunks = document.createElement("div");
  chunks.className = "hitPrev";
  const c0 = (item.chunks || []).slice(0, 2).map(c => `${c.source_ref}: ${c.preview}`).join("\n");
  chunks.textContent = c0 ? `Chunks:\n${c0}` : "";

  el.appendChild(top);
  el.appendChild(prev);
  el.appendChild(chunks);
  return el;
}

async function onGraphQuery() {
  const cfg = getCfg();
  const q = $("graphQuery").value.trim();
  if (!q) return;
  const resEl = $("graphResults");
  resEl.textContent = "Querying…";
  const url = `/api/graph/query?db_path=${encodeURIComponent(cfg.db_path)}&q=${encodeURIComponent(q)}&entity_limit=5`;
  const r = await fetch(url);
  const data = await r.json();
  if (!r.ok || data.ok === false) {
    resEl.textContent = data.error || "Graph query failed";
    return;
  }
  resEl.textContent = "";
  const ents = (data.result && data.result.entities) ? data.result.entities : [];
  for (const e of ents) resEl.appendChild(graphEntityCard(e));
}

window.addEventListener("DOMContentLoaded", () => {
  tabInit();
  refreshStatus();
  setInterval(refreshStatus, 8000);

  $("btnIngest").addEventListener("click", () => onIngest().catch(e => log(`Ingest error: ${e.message}`)));
  $("btnIndex").addEventListener("click", () => onIndex().catch(e => log(`Index error: ${e.message}`)));
  $("btnGraph").addEventListener("click", () => onGraphBuild().catch(e => log(`Graph error: ${e.message}`)));
  $("btnAsk").addEventListener("click", () => onAsk().catch(e => { $("answer").textContent = `Error: ${e.message}`; }));
  $("btnSearch").addEventListener("click", () => onSearch().catch(e => { $("searchResults").textContent = `Error: ${e.message}`; }));
  $("btnGraphQuery").addEventListener("click", () => onGraphQuery().catch(e => { $("graphResults").textContent = `Error: ${e.message}`; }));
});

