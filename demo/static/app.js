/* Live demo front-end: everything is drawn client-side on canvases from the probability maps
   returned by the local server, so the threshold slider and the view modes react instantly. */
"use strict";

const MODELS = ["U-Net", "DeepLabV3+"];
const COLORS = { gt: [22, 163, 74], "U-Net": [37, 99, 235], "DeepLabV3+": [234, 88, 12],
                 tp: [34, 197, 94], fp: [220, 38, 38], fn: [124, 58, 237] };
const GROUP_LABEL = { typical: "Typical", unet: "U-Net better", deeplab: "DeepLabV3+ better",
                      failure: "Failure", upload: "Upload" };
const GROUP_SHORT = { typical: "Typical", unet: "U-Net wins", deeplab: "DLV3+ wins", failure: "Failure", upload: "Upload" };
const DX_LABEL = { "Melanoma (all)": "Melanoma", "Nevus": "Nevus", "Seborrheic keratosis": "Seb. keratosis" };

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const fmt = (v, d = 3) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d));

const state = {
  info: null, list: [], current: null, result: null, pix: null,
  view: "overlay", threshold: 0.5, showGT: true, seeds: {}, group: "all", reqId: 0,
};

// ------------------------------------------------------------------ boot
async function boot() {
  try {
    const r = await fetch("/api/info");
    state.info = await r.json();
  } catch (e) {
    setStatus("err", "server not reachable");
    return;
  }
  const info = state.info;
  state.seeds = { ...info.default_seeds };
  state.list = info.examples.map((e) => ({ ...e }));
  const loaded = MODELS.filter((m) => info.available[m].length);
  setStatus(loaded.length ? "ok" : "err",
    loaded.length ? `${info.device.toUpperCase()} · ${loaded.join(" + ")} loaded` : "no checkpoints found");
  renderSeeds();
  renderKPIs();
  renderGallery();
  renderResults();
  if (state.list.length) select(state.list[0].id);
}

function setStatus(kind, text) {
  $("#status").className = "status " + kind;
  $("#status-text").textContent = text;
}

// ------------------------------------------------------------------ tabs & controls
$$(".tab").forEach((b) => b.addEventListener("click", () => {
  $$(".tab").forEach((x) => x.classList.toggle("active", x === b));
  $$(".tabpane").forEach((p) => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
}));

$$("#view button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
function setView(v) {
  state.view = v;
  $$("#view button").forEach((x) => x.classList.toggle("active", x.dataset.view === v));
  draw();
}

let rescoreTimer = null;
$("#thr").addEventListener("input", (e) => {
  state.threshold = parseFloat(e.target.value);
  $("#thr-val").textContent = state.threshold.toFixed(2);
  draw();
  clearTimeout(rescoreTimer);
  rescoreTimer = setTimeout(rescore, 120);
});
$("#show-gt").addEventListener("change", (e) => { state.showGT = e.target.checked; draw(); });

$$("#filters .chip").forEach((c) => c.addEventListener("click", () => {
  state.group = c.dataset.group;
  $$("#filters .chip").forEach((x) => x.classList.toggle("active", x === c));
  renderGallery();
}));

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" && e.target.type !== "range" && e.target.type !== "checkbox") return;
  if (e.target.tagName === "SELECT") return;
  if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    const vis = visibleList();
    const i = vis.findIndex((x) => x.id === state.current);
    if (!vis.length) return;
    const j = (i + (e.key === "ArrowRight" ? 1 : -1) + vis.length) % vis.length;
    select(vis[j].id);
    e.preventDefault();
  }
  const views = { 1: "overlay", 2: "contour", 3: "prob", 4: "error" };
  if (views[e.key]) setView(views[e.key]);
  if (e.key === "Escape") $("#lightbox").hidden = true;
});

// ------------------------------------------------------------------ upload
const drop = $("#drop");
$("#file").addEventListener("change", (e) => e.target.files[0] && upload(e.target.files[0]));
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));

function upload(file) {
  const reader = new FileReader();
  reader.onload = () => {
    const id = "upload-" + Date.now();
    state.list.unshift({ id, group: "upload", upload: true, image: reader.result, name: file.name,
                         note: "Your image: no ground truth, so only the predictions and the predicted area are shown." });
    state.group = "all";
    $$("#filters .chip").forEach((x) => x.classList.toggle("active", x.dataset.group === "all"));
    renderGallery();
    select(id);
  };
  reader.readAsDataURL(file);
}

// ------------------------------------------------------------------ gallery
function visibleList() {
  return state.list.filter((e) => state.group === "all" || e.group === state.group);
}

function renderGallery() {
  const g = $("#gallery");
  g.innerHTML = "";
  for (const e of visibleList()) {
    const b = document.createElement("button");
    b.className = "thumb" + (e.id === state.current ? " selected" : "");
    b.dataset.id = e.id;
    const src = e.upload ? e.image : `/demo/examples/${e.id}.jpg`;
    const cap = e.upload ? `<span>${escapeHtml(e.name || "upload")}</span>`
                         : `<span>${DX_LABEL[e.diagnosis] || e.diagnosis}</span><span>${e.area_pct}%</span>`;
    b.innerHTML = `<img src="${src}" alt="${e.id}" loading="lazy"><span class="t-tag tag-${e.group}">${GROUP_SHORT[e.group]}</span><div class="t-cap">${cap}</div>`;
    b.addEventListener("click", () => select(e.id));
    g.appendChild(b);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ------------------------------------------------------------------ seeds
function renderSeeds() {
  for (const m of MODELS) {
    const box = $(`.seed-sel[data-model="${CSS.escape(m)}"]`);
    const seeds = state.info.available[m];
    box.innerHTML = "";
    if (!seeds.length) continue;
    const sel = document.createElement("select");
    sel.title = "Training run (random seed) of this model";
    for (const s of seeds) sel.add(new Option("seed " + s, s, false, s === state.seeds[m]));
    sel.addEventListener("change", () => { state.seeds[m] = parseInt(sel.value, 10); if (state.current) select(state.current, true); });
    box.appendChild(sel);
  }
}

// ------------------------------------------------------------------ predict
async function select(id, force = false) {
  if (id === state.current && !force && state.result) return;
  state.current = id;
  $$(".thumb").forEach((t) => t.classList.toggle("selected", t.dataset.id === id));
  const sel = $(`.thumb[data-id="${CSS.escape(id)}"]`);
  if (sel) sel.scrollIntoView({ block: "nearest" });
  const e = state.list.find((x) => x.id === id);
  renderCaseHead(e);
  setLoading(true);
  const myReq = ++state.reqId;
  const body = { seeds: state.seeds, threshold: state.threshold };
  if (e.upload) body.image = e.image; else body.example = e.id;
  try {
    const r = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const res = await r.json();
    if (!r.ok) throw new Error(res.error || r.statusText);
    if (myReq !== state.reqId) return;
    state.result = res;
    state.pix = await loadPixels(res);
    if (myReq !== state.reqId) return;
    $$(".canvas-wrap").forEach((el) => (el.style.aspectRatio = `${res.width} / ${res.height}`));
    showError(null);
    draw();
    renderMetrics();
  } catch (err) {
    if (myReq === state.reqId) showError(err.message);
  } finally {
    if (myReq === state.reqId) setLoading(false);
  }
}

async function rescore() {
  if (!state.result) return;
  const r = await fetch("/api/rescore", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key: state.result.key, threshold: state.threshold }) });
  const res = await r.json();
  if (!r.ok) return;
  for (const m of Object.keys(res.models)) state.result.models[m].metrics = res.models[m].metrics;
  state.result.agreement = res.agreement;
  state.result.threshold = res.threshold;
  renderMetrics();
}

function setLoading(on) { $$(".spinner").forEach((s) => (s.hidden = !on)); }
function showError(msg) { const b = $("#error"); b.hidden = !msg; b.textContent = msg ? "Error: " + msg : ""; }

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error("could not load " + src.slice(0, 60)));
    im.src = src;
  });
}

function pixelsOf(im, w, h) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(im, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h).data;
}

async function loadPixels(res) {
  const w = res.width, h = res.height;
  const img = await loadImage(res.image);
  const out = { w, h, rgb: pixelsOf(img, w, h), gt: null, prob: {} };
  if (res.gt) {
    const d = pixelsOf(await loadImage(res.gt), w, h);
    out.gt = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) out.gt[i] = d[i * 4] > 127 ? 1 : 0;
  }
  for (const m of Object.keys(res.models)) {
    const d = pixelsOf(await loadImage(res.models[m].prob), w, h);
    const p = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) p[i] = d[i * 4];
    out.prob[m] = p;
  }
  return out;
}

// ------------------------------------------------------------------ drawing
function binarize(prob, t) {
  const n = prob.length, m = new Uint8Array(n), thr = t * 255;
  for (let i = 0; i < n; i++) m[i] = prob[i] > thr ? 1 : 0;
  return m;
}

function edges(mask, w, h, r) {
  // boundary band: pixels whose (2r+1)x(2r+1) neighbourhood contains both classes
  const e = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = mask[y * w + x];
      let diff = false;
      for (let dy = -r; dy <= r && !diff; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -r; dx <= r; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          if (mask[yy * w + xx] !== v) { diff = true; break; }
        }
      }
      if (diff) e[y * w + x] = 1;
    }
  }
  return e;
}

const INFERNO = [[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]];
function cmap(v) {
  const x = Math.min(1, Math.max(0, v)) * (INFERNO.length - 1);
  const i = Math.min(INFERNO.length - 2, Math.floor(x)), f = x - i;
  const a = INFERNO[i], b = INFERNO[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

function paint(canvas, fn) {
  const { w, h } = state.pix;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d");
  const out = ctx.createImageData(w, h);
  fn(out.data);
  ctx.putImageData(out, 0, 0);
}

function blend(dst, i, c, a) {
  dst[i] = dst[i] * (1 - a) + c[0] * a;
  dst[i + 1] = dst[i + 1] * (1 - a) + c[1] * a;
  dst[i + 2] = dst[i + 2] * (1 - a) + c[2] * a;
}

function draw() {
  if (!state.pix || !state.result) return;
  const { w, h, rgb, gt } = state.pix;
  const r = Math.max(1, Math.round(w / 400));
  const gtEdge = gt ? edges(gt, w, h, r) : null;
  const showGT = state.showGT && gt;
  $("#show-gt").disabled = !gt;
  $("#show-gt").parentElement.classList.toggle("disabled", !gt);
  $("#gt-label").textContent = gt ? (state.showGT ? "+ ground truth" : "") : "(no ground truth)";

  paint($("#cv-input"), (d) => {
    d.set(rgb);
    if (showGT) for (let i = 0; i < w * h; i++) if (gtEdge[i]) blend(d, i * 4, COLORS.gt, 1);
  });

  for (const m of MODELS) {
    const canvas = document.getElementById("cv-" + m);
    const panel = document.getElementById("panel-" + m);
    if (!state.pix.prob[m]) { panel.style.opacity = 0.4; canvas.width = 1; canvas.height = 1; continue; }
    panel.style.opacity = 1;
    const prob = state.pix.prob[m];
    const mask = binarize(prob, state.threshold);
    const pe = edges(mask, w, h, r);
    const col = COLORS[m];
    const view = state.view === "error" && !gt ? "overlay" : state.view;
    paint(canvas, (d) => {
      d.set(rgb);
      for (let i = 0; i < w * h; i++) {
        const k = i * 4;
        if (view === "overlay") {
          if (mask[i]) blend(d, k, col, 0.38);
          if (pe[i]) blend(d, k, col, 1);
        } else if (view === "contour") {
          if (pe[i]) blend(d, k, col, 1);
        } else if (view === "prob") {
          const p = prob[i] / 255;
          blend(d, k, cmap(p), 0.25 + 0.6 * p);
          if (pe[i]) blend(d, k, [255, 255, 255], 0.9);
        } else if (view === "error") {
          blend(d, k, [0, 0, 0], 0.35);
          if (mask[i] && gt[i]) blend(d, k, COLORS.tp, 0.55);
          else if (mask[i]) blend(d, k, COLORS.fp, 0.7);
          else if (gt[i]) blend(d, k, COLORS.fn, 0.7);
        }
        if (showGT && view !== "error" && gtEdge[i]) blend(d, k, COLORS.gt, 1);
      }
    });
  }
  renderLegend();
}

function renderLegend() {
  const sw = (c, t) => `<span><i style="background:rgb(${c.join(",")})"></i>${t}</span>`;
  let html = "";
  const gt = state.pix && state.pix.gt;
  if (state.view === "error" && gt) {
    html = sw(COLORS.tp, "correct (TP)") + sw(COLORS.fp, "extra (FP)") + sw(COLORS.fn, "missed (FN)");
  } else if (state.view === "prob") {
    html = `<span>probability</span><span><i style="width:80px;background:linear-gradient(90deg,${INFERNO.map((c) => `rgb(${c.join(",")})`).join(",")})"></i>0 → 1</span>` + sw([255, 255, 255], "threshold contour");
    if (gt && state.showGT) html += sw(COLORS.gt, "ground truth");
  } else {
    if (gt && state.showGT) html += sw(COLORS.gt, "ground truth");
    html += sw(COLORS["U-Net"], "U-Net") + sw(COLORS["DeepLabV3+"], "DeepLabV3+");
  }
  $("#legend").innerHTML = html;
}

// ------------------------------------------------------------------ metrics
function renderCaseHead(e) {
  if (e.upload) {
    $("#case-title").textContent = e.name || "Uploaded image";
    $("#case-meta").innerHTML = `<span class="pill">your image</span>no ground truth`;
  } else {
    $("#case-title").textContent = e.id;
    $("#case-meta").innerHTML = `<span class="pill grp tag-${e.group}">${GROUP_LABEL[e.group]}</span>` +
      `<span class="pill">${e.diagnosis}</span><span class="pill">lesion ${e.area_pct}% of image</span>` +
      `<span class="pill">original ${e.original_size[0]}×${e.original_size[1]}</span>`;
  }
  $("#case-note").textContent = e.note || "";
}

function renderMetrics() {
  const res = state.result;
  const hasGT = !!state.pix.gt;
  const ms = MODELS.filter((m) => res.models[m]);
  const best = {};
  if (ms.length === 2 && hasGT) {
    const [a, b] = ms.map((m) => res.models[m].metrics);
    best.dice = Math.abs(a.dice - b.dice) < 0.005 ? null : (a.dice > b.dice ? ms[0] : ms[1]);
    best.iou = Math.abs(a.iou - b.iou) < 0.005 ? null : (a.iou > b.iou ? ms[0] : ms[1]);
    best.hd95 = Math.abs(a.hd95 - b.hd95) < 0.5 ? null : (a.hd95 < b.hd95 ? ms[0] : ms[1]);
  }
  for (const m of MODELS) {
    const box = $(`.metrics[data-model="${m}"]`);
    const panel = document.getElementById("panel-" + m);
    const info = res.models[m];
    panel.classList.toggle("best", !!best.dice && best.dice === m);
    $(".badge", panel).hidden = !(best.dice && best.dice === m);
    $(".seed-tag", panel).textContent = info ? `${info.ms.toFixed(0)} ms` : "checkpoint missing";
    if (!info) { box.innerHTML = `<div class="metric"><div class="m-label">no checkpoint</div><div class="m-val">—</div></div>`; continue; }
    const mt = info.metrics;
    const col = `rgb(${COLORS[m].join(",")})`;
    const cell = (label, val, win, bar) =>
      `<div class="metric"><div class="m-label">${label}</div><div class="m-val${win ? " win" : ""}">${val}</div>` +
      (bar != null ? `<div class="m-bar"><span style="width:${Math.max(0, Math.min(1, bar)) * 100}%;background:${col}"></span></div>` : "") + `</div>`;
    if (hasGT) {
      box.innerHTML = cell("Dice", fmt(mt.dice), best.dice === m, mt.dice) +
        cell("IoU", fmt(mt.iou), best.iou === m, mt.iou) +
        cell("HD95 px", fmt(mt.hd95, 1), best.hd95 === m, null) +
        cell("Area", mt.area_pct.toFixed(1) + "%", false, null);
    } else {
      box.innerHTML = cell("Area", mt.area_pct.toFixed(1) + "%", false, mt.area_pct / 100) +
        cell("Time, ms", info.ms.toFixed(0), false, null) +
        cell("Epoch", info.epoch ?? "—", false, null) +
        cell("Params, M", info.params_m.toFixed(1), false, null);
    }
  }
  renderAgreement();
  renderSeedTable();
}

function renderAgreement() {
  const res = state.result;
  const a = res.agreement;
  const e = state.list.find((x) => x.id === state.current);
  let html = `<h3>Do the two models agree?</h3>`;
  if (!a) { $("#agreement").innerHTML = html + `<span class="muted">needs both checkpoints</span>`; return; }
  const verdict = a.dice > 0.95 ? "almost identical masks" : a.dice > 0.85 ? "similar masks, border differs" : "clearly different masks";
  html += `<div style="font-size:22px;font-weight:700">Dice ${fmt(a.dice)} <span class="muted" style="font-size:13px">${verdict}</span></div>`;
  html += `<div class="muted">IoU ${fmt(a.iou)} · HD95 ${fmt(a.hd95, 1)} px between U-Net and DeepLabV3+ masks at threshold ${res.threshold.toFixed(2)}`;
  if (e && !e.upload) html += ` · ground-truth area ${e.area_pct}%`;
  $("#agreement").innerHTML = html + `</div>`;
}

function renderSeedTable() {
  const e = state.list.find((x) => x.id === state.current);
  const box = $("#seed-table");
  if (!e || e.upload || !e.eval_dice) {
    box.innerHTML = `<h3>Same image in every training run</h3><span class="muted">Available for the validation examples (they have ground truth).</span>`;
    return;
  }
  const seeds = ["1", "2", "42"];
  let html = `<h3>Same image in every training run <span class="muted">(Dice from evaluate.py, threshold 0.5)</span></h3><table class="mini-table"><thead><tr><th>Model</th>`;
  html += seeds.map((s) => `<th>seed ${s}</th>`).join("") + `<th>mean</th></tr></thead><tbody>`;
  for (const m of MODELS) {
    const d = e.eval_dice[m] || {};
    const vals = seeds.map((s) => d[s]).filter((v) => v != null);
    const mean = vals.length ? vals.reduce((x, y) => x + y, 0) / vals.length : null;
    html += `<tr><td style="color:rgb(${COLORS[m].join(",")});font-weight:600">${m}</td>` +
      seeds.map((s) => `<td>${fmt(d[s])}</td>`).join("") + `<td><b>${fmt(mean)}</b></td></tr>`;
  }
  box.innerHTML = html + `</tbody></table>`;
}

// ------------------------------------------------------------------ KPIs & results tab
function agg() { return (state.info.summary || {}).aggregate; }

function renderKPIs() {
  const a = agg();
  const sp = (state.info.summary || {}).speed;
  if (!a) { $("#kpis").hidden = true; return; }
  const u = a["U-Net"], d = a["DeepLabV3+"];
  const k = (label, uv, dv, unit = "") =>
    `<div class="kpi"><div class="k-label">${label}</div><div class="k-vals">` +
    `<div class="k-val unet">${uv}${unit}<small> U-Net</small></div><div class="k-val dlv3">${dv}${unit}<small> DeepLabV3+</small></div></div></div>`;
  let html = k("Validation Dice (3 seeds)", fmt(u.dice.mean), fmt(d.dice.mean)) +
    k("Worst-case border error, HD95 p95", fmt(u.hd95_p95.mean, 1), fmt(d.hd95_p95.mean, 1), " px") +
    k("Failed images (IoU < 0.65)", fmt(u.below.mean, 1), fmt(d.below.mean, 1));
  if (sp) html += k(`Inference per image (${sp.chip || "MPS"})`, fmt(sp["U-Net"].infer_ms, 1), fmt(sp["DeepLabV3+"].infer_ms, 1), " ms");
  $("#kpis").innerHTML = html;
}

function pm(s, d = 3) { return `${fmt(s.mean, d)} <span class="pm">± ${fmt(s.std, d)}</span>`; }

function renderResults() {
  const s = state.info.summary || {};
  const a = s.aggregate;
  if (!a) return;
  const u = a["U-Net"], d = a["DeepLabV3+"];
  const sp = s.speed, params = s.params_m || {};
  const rows = [
    ["Dice", "dice", 3, true], ["IoU", "iou", 3, true], ["ISIC thresholded Jaccard", "isic", 3, true],
    ["Images with IoU < 0.65", "below", 1, false], ["HD95 median, px", "hd95_med", 2, false],
    ["HD95 95th percentile, px", "hd95_p95", 1, false], ["Training time, min", "train_min", 1, false],
  ];
  let html = `<table class="data"><thead><tr><th>Metric</th><th class="head-unet">U-Net</th><th class="head-dlv3">DeepLabV3+</th><th>Difference</th></tr></thead><tbody>`;
  for (const [label, key, dec, higher] of rows) {
    const x = u[key], y = d[key];
    const diff = y.mean - x.mean;
    const noise = 2 * Math.max(x.std, y.std);  // "clear" only beyond twice the seed std
    const clear = Math.abs(diff) > noise;
    const uw = clear && (higher ? x.mean > y.mean : x.mean < y.mean);
    const dw = clear && !uw;
    html += `<tr><td>${label}</td><td class="${uw ? "win" : ""}">${pm(x, dec)}</td><td class="${dw ? "win" : ""}">${pm(y, dec)}</td>` +
      `<td>${diff >= 0 ? "+" : ""}${diff.toFixed(dec)} <span class="pm">${clear ? "" : "≈ seed noise"}</span></td></tr>`;
  }
  if (params["U-Net"]) html += `<tr><td>Parameters, M</td><td>${params["U-Net"].toFixed(1)}</td><td>${params["DeepLabV3+"].toFixed(1)}</td><td></td></tr>`;
  if (sp) {
    const r = sp["DeepLabV3+"].infer_ms / sp["U-Net"].infer_ms;
    html += `<tr><td>Inference, ms / image (${sp.chip})</td><td class="win">${sp["U-Net"].infer_ms.toFixed(1)}</td><td>${sp["DeepLabV3+"].infer_ms.toFixed(1)}</td><td>DeepLabV3+ ${r.toFixed(1)}× slower</td></tr>`;
    html += `<tr><td>Epoch time, s (${sp.chip})</td><td class="win">${sp["U-Net"].epoch_s.toFixed(0)}</td><td>${sp["DeepLabV3+"].epoch_s.toFixed(0)}</td><td></td></tr>`;
  }
  $("#main-table").innerHTML = html + `</tbody></table>`;
  const p = s.paired;
  if (p) {
    $("#paired-note").innerHTML = `Paired per-image test (Dice averaged over seeds, DeepLabV3+ − U-Net): mean ${p.mean_diff.toFixed(4)}, ` +
      `95% bootstrap CI ${p.ci95[0].toFixed(4)} … +${p.ci95[1].toFixed(4)}, Wilcoxon p = ${p.wilcoxon_p.toFixed(3)}; ` +
      `U-Net better on ${p.unet_better} images, DeepLabV3+ on ${p.dlv3p_better}. Green = better by more than twice the seed std.`;
  }
  renderDotplot(s.runs);
  renderGroupTable("#by-size", s.by_size, "Lesion area");
  renderGroupTable("#by-dx", s.by_dx, "Diagnosis");
  renderFindings(s);
  renderAblation(s.ablation || []);
  renderFigures(s.figures || []);
}

function renderDotplot(runs) {
  if (!runs) return;
  const vals = runs.map((r) => r.dice);
  const lo = Math.floor((Math.min(...vals) - 0.002) * 500) / 500, hi = Math.ceil((Math.max(...vals) + 0.002) * 500) / 500;
  const W = 520, H = 150, L = 92, R = 16;
  const x = (v) => L + (v - lo) / (hi - lo) * (W - L - R);
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Dice per run">`;
  for (let t = lo; t <= hi + 1e-9; t += 0.002) {
    svg += `<line x1="${x(t)}" x2="${x(t)}" y1="14" y2="${H - 28}" stroke="#e5e7eb"/><text x="${x(t)}" y="${H - 12}" font-size="11" text-anchor="middle" fill="#6b7280">${t.toFixed(3)}</text>`;
  }
  MODELS.forEach((m, i) => {
    const y = 40 + i * 50;
    const rr = runs.filter((r) => r.model === m);
    const mean = rr.reduce((s, r) => s + r.dice, 0) / rr.length;
    const c = `rgb(${COLORS[m].join(",")})`;
    svg += `<text x="0" y="${y + 4}" font-size="12.5" font-weight="600" fill="${c}">${m}</text>`;
    svg += `<line x1="${x(mean)}" x2="${x(mean)}" y1="${y - 14}" y2="${y + 14}" stroke="${c}" stroke-width="2.5"/>`;
    for (const r of rr) {
      svg += `<circle cx="${x(r.dice)}" cy="${y}" r="6.5" fill="${c}" fill-opacity=".75" stroke="#fff" stroke-width="1.5"><title>seed ${r.seed}: ${r.dice.toFixed(4)}</title></circle>`;
      svg += `<text x="${x(r.dice)}" y="${y - 11}" font-size="10" text-anchor="middle" fill="#6b7280">s${r.seed}</text>`;
    }
  });
  $("#dotplot").innerHTML = svg + `</svg>`;
}

function renderGroupTable(sel, t, label) {
  if (!t) return;
  let html = `<table class="data"><thead><tr><th>${label}</th><th>n</th><th class="head-unet">U-Net</th><th class="head-dlv3">DeepLabV3+</th><th>Δ (DLV3+ − U-Net)</th></tr></thead><tbody>`;
  for (const [k, v] of Object.entries(t)) {
    const u = v["U-Net"], d = v["DeepLabV3+"];
    if (!u || !d) continue;
    const diff = d.mean - u.mean;
    html += `<tr><td>${k}</td><td>${v.n}</td><td>${pm(u)}</td><td>${pm(d)}</td><td><span class="delta ${diff > 0 ? "pos" : "neg"}">${diff >= 0 ? "+" : ""}${diff.toFixed(3)}</span></td></tr>`;
  }
  $(sel).innerHTML = html + `</tbody></table>`;
}

function renderFindings(s) {
  const a = s.aggregate, u = a["U-Net"], d = a["DeepLabV3+"];
  const small = s.by_size && s.by_size["< 5%"];
  const sp = s.speed;
  const items = [
    `<b>Both models reach Dice ≈ 0.90</b> on the validation split (U-Net ${fmt(u.dice.mean)}, DeepLabV3+ ${fmt(d.dice.mean)}). ` +
      `The gap (${(d.dice.mean - u.dice.mean).toFixed(4)}) is about one seed standard deviation (${fmt(Math.max(u.dice.std, d.dice.std), 4)}).`,
    `<b>DeepLabV3+ tends to make smaller worst-case border errors</b>: HD95 95th percentile ${fmt(d.hd95_p95.mean, 1)} vs ${fmt(u.hd95_p95.mean, 1)} px (lower in 2 of 3 seeds), ` +
      `but it does not fail on fewer images (${fmt(d.below.mean, 1)} vs ${fmt(u.below.mean, 1)} with IoU < 0.65).`,
  ];
  if (small && small["U-Net"]) items.push(`<b>U-Net is better on small lesions</b> (< 5% of the image: ${fmt(small["U-Net"].mean)} vs ${fmt(small["DeepLabV3+"].mean)}): ` +
      `skip connections at every scale keep the fine border, DeepLabV3+ upsamples a 1/4-resolution mask.`);
  if (sp) items.push(`<b>U-Net is ${(sp["DeepLabV3+"].infer_ms / sp["U-Net"].infer_ms).toFixed(1)}× faster</b> at inference (${sp["U-Net"].infer_ms.toFixed(1)} vs ${sp["DeepLabV3+"].infer_ms.toFixed(1)} ms per image).`);
  items.push(`<b>Failures are shared and data-driven</b>: ambiguous annotations, faint non-pigmented lesions, and artefacts such as immersion-fluid bubbles (see the Failures examples).`);
  $("#findings").innerHTML = items.map((x) => `<li>${x}</li>`).join("");
}

function renderAblation(list) {
  if (!list.length) return;
  $("#ablation-card").hidden = false;
  let html = "";
  for (const { name, data } of list) {
    html += `<h3 style="font-size:13px;margin:8px 0 6px">${escapeHtml(name.replace(/_/g, " "))}</h3>`;
    const rows = Array.isArray(data) ? data : (data.rows || data.results || Object.entries(data).map(([k, v]) => (typeof v === "object" ? { setting: k, ...v } : { setting: k, value: v })));
    if (!Array.isArray(rows) || !rows.length || typeof rows[0] !== "object") { html += `<pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre>`; continue; }
    const cols = [...new Set(rows.flatMap((r) => Object.keys(r)))].filter((c) => typeof rows[0][c] !== "object");
    html += `<table class="data"><thead><tr>${cols.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr></thead><tbody>`;
    for (const r of rows) html += `<tr>${cols.map((c) => `<td>${typeof r[c] === "number" ? (Number.isInteger(r[c]) ? r[c] : r[c].toFixed(4)) : escapeHtml(r[c] ?? "")}</td>`).join("")}</tr>`;
    html += `</tbody></table>`;
  }
  $("#ablation").innerHTML = html;
}

function renderFigures(figs) {
  $("#figs").innerHTML = figs.map((f) => `<figure class="fig" data-src="${f.src}" data-cap="${escapeHtml(f.caption)}"><img src="${f.src}" alt="${escapeHtml(f.caption)}" loading="lazy"><figcaption>${escapeHtml(f.caption)}</figcaption></figure>`).join("");
  $$(".fig").forEach((f) => f.addEventListener("click", () => {
    const lb = $("#lightbox");
    $("img", lb).src = f.dataset.src;
    $(".lb-cap", lb).textContent = f.dataset.cap;
    lb.hidden = false;
  }));
}
$("#lightbox").addEventListener("click", () => ($("#lightbox").hidden = true));

boot();
