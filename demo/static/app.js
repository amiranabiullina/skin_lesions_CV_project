/* Minimal live demo: pick or drop an image, see both models' masks, scores and the overall results. */
"use strict";

const MODELS = ["U-Net", "DeepLabV3+"];
const RGB = { gt: [22, 163, 74], "U-Net": [37, 99, 235], "DeepLabV3+": [234, 88, 12],
              tp: [34, 197, 94], fp: [220, 38, 38], fn: [124, 58, 237] };
const css = (c) => `rgb(${c.join(",")})`;
const LABEL = { typical: "typical", unet: "U-Net wins", deeplab: "DLV3+ wins", failure: "failure" };
const DX = { "Melanoma (all)": "melanoma", "Nevus": "nevus", "Seborrheic keratosis": "seb. keratosis" };

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const f = (v, d = 3) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const state = { info: null, result: null, pix: null, view: "mask", req: 0, current: null };

// ------------------------------------------------------------------ boot
async function boot() {
  try {
    state.info = await (await fetch("/api/info")).json();
  } catch (e) { return setStatus("err", "server not running"); }
  const loaded = MODELS.filter((m) => state.info.available[m].length);
  setStatus(loaded.length === 2 ? "ok" : "err", loaded.length ? `models ready · ${state.info.device.toUpperCase()}` : "no checkpoints");
  renderSamples();
  renderResults();
  renderFailures();
}
function setStatus(kind, text) { $("#status").className = "status " + kind; $("#status-text").textContent = text; }

// ------------------------------------------------------------------ input
function renderSamples() {
  // one or two per group keeps the row short; every image is in demo/test_images as well
  const pick = ["ISIC_0000145", "ISIC_0014652", "ISIC_0013333", "ISIC_0012661", "ISIC_0014931", "ISIC_0013966", "ISIC_0013567"];
  const ex = state.info.examples;
  const list = pick.map((id) => ex.find((e) => e.id === id)).filter(Boolean);
  $("#samples").innerHTML = list.map((e) =>
    `<button class="sample" data-id="${e.id}" title="${e.id}: ${e.note}"><img src="/demo/examples/${e.id}.jpg" alt=""><span>${LABEL[e.group]}</span></button>`).join("");
  $$(".sample").forEach((b) => b.addEventListener("click", () => run({ example: b.dataset.id })));
}

const drop = $("#drop");
$("#file").addEventListener("change", (e) => e.target.files[0] && readFile(e.target.files[0]));
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && readFile(e.dataTransfer.files[0]));
function readFile(file) {
  const r = new FileReader();
  r.onload = () => run({ image: r.result, name: file.name });
  r.readAsDataURL(file);
}

$$("#view button").forEach((b) => b.addEventListener("click", () => {
  state.view = b.dataset.view;
  $$("#view button").forEach((x) => x.classList.toggle("active", x === b));
  draw();
}));

// ------------------------------------------------------------------ predict
async function run(input) {
  const my = ++state.req;
  $("#output").hidden = false;
  $$(".spin").forEach((s) => (s.hidden = false));
  $("#error").hidden = true;
  $$(".sample").forEach((b) => b.classList.toggle("on", b.dataset.id === input.example));
  $("#output").scrollIntoView({ behavior: "smooth", block: "start" });
  try {
    const body = { seeds: state.info.default_seeds, threshold: 0.5 };
    if (input.example) body.example = input.example; else body.image = input.image;
    const r = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const res = await r.json();
    if (!r.ok) throw new Error(res.error || r.statusText);
    const pix = await loadPixels(res);
    if (my !== state.req) return;
    state.result = res; state.pix = pix;
    state.current = res.example ? state.info.examples.find((e) => e.id === res.example) : { name: input.name };
    renderHead();
    draw();
    renderImageCharts();
  } catch (err) {
    if (my === state.req) { $("#error").hidden = false; $("#error").textContent = "Error: " + err.message; }
  } finally {
    if (my === state.req) $$(".spin").forEach((s) => (s.hidden = true));
  }
}

function renderHead() {
  const e = state.current;
  if (e.id) {
    $("#out-title").textContent = `${e.id} · ${DX[e.diagnosis] || e.diagnosis}, ${e.area_pct}% of the image`;
    $("#out-note").textContent = e.note;
  } else {
    $("#out-title").textContent = e.name || "Your image";
    $("#out-note").textContent = "No ground truth for this image, so the page shows the predictions and their area.";
  }
}

function loadImage(src) {
  return new Promise((ok, bad) => { const im = new Image(); im.onload = () => ok(im); im.onerror = () => bad(new Error("image failed to load")); im.src = src; });
}
function pixelsOf(im, w, h) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const x = c.getContext("2d", { willReadFrequently: true }); x.drawImage(im, 0, 0, w, h);
  return x.getImageData(0, 0, w, h).data;
}
async function loadPixels(res) {
  const w = res.width, h = res.height, n = w * h;
  const out = { w, h, rgb: pixelsOf(await loadImage(res.image), w, h), gt: null, mask: {} };
  if (res.gt) {
    const d = pixelsOf(await loadImage(res.gt), w, h);
    out.gt = new Uint8Array(n); for (let i = 0; i < n; i++) out.gt[i] = d[4 * i] > 127;
  }
  for (const m of Object.keys(res.models)) {
    const d = pixelsOf(await loadImage(res.models[m].prob), w, h);
    const mk = new Uint8Array(n); for (let i = 0; i < n; i++) mk[i] = d[4 * i] > 127;  // threshold 0.5
    out.mask[m] = mk;
  }
  return out;
}

// ------------------------------------------------------------------ drawing
function edges(mask, w, h, r) {
  const e = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const v = mask[y * w + x];
    outer: for (let dy = -r; dy <= r; dy++) {
      const yy = y + dy; if (yy < 0 || yy >= h) continue;
      for (let dx = -r; dx <= r; dx++) {
        const xx = x + dx; if (xx < 0 || xx >= w) continue;
        if (mask[yy * w + xx] !== v) { e[y * w + x] = 1; break outer; }
      }
    }
  }
  return e;
}
function blend(d, k, c, a) { d[k] += (c[0] - d[k]) * a; d[k + 1] += (c[1] - d[k + 1]) * a; d[k + 2] += (c[2] - d[k + 2]) * a; }
function paint(canvas, fn) {
  const { w, h, rgb } = state.pix;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(w, h);
  img.data.set(rgb); fn(img.data); ctx.putImageData(img, 0, 0);
}

function draw() {
  if (!state.pix) return;
  const { w, h, gt, mask } = state.pix, n = w * h, r = Math.max(1, Math.round(w / 320));
  const ge = gt ? edges(gt, w, h, r) : null;
  const errorView = state.view === "error" && gt;
  $("#cap-input").textContent = gt ? "Image + ground truth" : "Your image";
  paint($("#cv-input"), (d) => { if (ge) for (let i = 0; i < n; i++) if (ge[i]) blend(d, 4 * i, RGB.gt, 1); });
  for (const m of MODELS) {
    const cv = document.getElementById("cv-" + m), mk = mask[m];
    if (!mk) { cv.width = 1; cv.height = 1; continue; }
    const pe = edges(mk, w, h, r), c = RGB[m];
    paint(cv, (d) => {
      for (let i = 0; i < n; i++) {
        const k = 4 * i;
        if (errorView) {
          blend(d, k, [0, 0, 0], 0.3);
          if (mk[i] && gt[i]) blend(d, k, RGB.tp, 0.55);
          else if (mk[i]) blend(d, k, RGB.fp, 0.7);
          else if (gt[i]) blend(d, k, RGB.fn, 0.7);
        } else {
          if (mk[i]) blend(d, k, c, 0.28);
          if (ge && ge[i]) blend(d, k, RGB.gt, 1);
          if (pe[i]) blend(d, k, c, 1);
        }
      }
    });
  }
  const sw = (c, t) => `<span><i style="background:${css(c)}"></i>${t}</span>`;
  $("#legend").innerHTML = errorView
    ? sw(RGB.tp, "correct") + sw(RGB.fp, "extra (false positive)") + sw(RGB.fn, "missed (false negative)")
    : (gt ? sw(RGB.gt, "ground truth (expert)") : "") + sw(RGB["U-Net"], "U-Net") + sw(RGB["DeepLabV3+"], "DeepLabV3+");
  $("#view").style.visibility = gt ? "visible" : "hidden";
}

// ------------------------------------------------------------------ charts
function hbars(rows, max, fmtv) {
  // rows: [{label, value, color}] -> horizontal bar chart with value labels
  const W = 460, rowH = 30, L = 120, R = 56, H = rows.length * rowH + 6;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
  rows.forEach((r, i) => {
    const y = 4 + i * rowH, bw = Math.max(2, (W - L - R) * Math.min(1, r.value / max));
    if (r.group) s += `<text x="0" y="${y + 17}" font-size="13" fill="#15181e" font-weight="600">${r.group}</text>`;
    s += `<text x="${L - 8}" y="${y + 17}" font-size="12.5" fill="#6b7280" text-anchor="end">${r.label}</text>`;
    s += `<rect x="${L}" y="${y + 5}" width="${W - L - R}" height="16" rx="4" fill="#e9ebef"/>`;
    s += `<rect x="${L}" y="${y + 5}" width="${bw}" height="16" rx="4" fill="${r.color}"/>`;
    s += `<text x="${L + bw + 6}" y="${y + 17}" font-size="12.5" font-weight="700" fill="#15181e">${fmtv(r.value)}</text>`;
  });
  return s + "</svg>";
}

function renderImageCharts() {
  const res = state.result, gt = !!state.pix.gt;
  for (const m of MODELS) {
    const info = res.models[m], el = document.getElementById("score-" + m);
    if (!info) { el.textContent = "no checkpoint"; continue; }
    el.innerHTML = (gt ? `Dice ${f(info.metrics.dice)}` : `${info.metrics.area_pct.toFixed(1)}%`) + `<small>${info.ms.toFixed(0)} ms</small>`;
  }
  const ms = MODELS.filter((m) => res.models[m]);
  if (gt) {
    const rows = [];
    for (const [key, name] of [["dice", "Dice"], ["iou", "IoU"]]) {
      ms.forEach((m, i) => rows.push({ group: i === 0 ? name : "", label: m, value: res.models[m].metrics[key], color: css(RGB[m]) }));
    }
    const hd = ms.map((m) => `<b style="color:${css(RGB[m])}">${f(res.models[m].metrics.hd95, 1)} px</b>`).join(" vs ");
    $("#chart-image").innerHTML = hbars(rows, 1, (v) => v.toFixed(3)) +
      `<p class="sub" style="margin:8px 0 0">Border error HD95: ${hd} (lower is better)</p>`;
  } else {
    $("#chart-image").innerHTML = `<div class="empty">Scores need a ground-truth mask. Drop an image from <code>demo/test_images</code> to see Dice and IoU.</div>`;
  }
  const area = [];
  if (gt) area.push({ label: "ground truth", value: 100 * mean(state.pix.gt), color: css(RGB.gt) });
  ms.forEach((m) => area.push({ label: m, value: res.models[m].metrics.area_pct, color: css(RGB[m]) }));
  const top = Math.max(10, ...area.map((a) => a.value)) * 1.1;
  $("#chart-area").innerHTML = hbars(area, top, (v) => v.toFixed(1) + "%");
}
function mean(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; }

// ------------------------------------------------------------------ overall results
function renderResults() {
  const s = state.info.summary || {}, a = s.aggregate;
  if (!a) { $("#results").hidden = true; return; }
  const u = a["U-Net"], d = a["DeepLabV3+"], sp = s.speed;
  const tile = (label, uv, dv, unit = "") => `<div class="stat"><div class="label">${label}</div><div class="pair">` +
    `<div class="v c-unet">${uv}<small>${unit}</small></div><div class="v c-dlv3">${dv}<small>${unit}</small></div></div></div>`;
  $("#stats").innerHTML =
    tile("Dice (higher is better)", f(u.dice.mean), f(d.dice.mean)) +
    tile("Worst border errors, HD95 p95", f(u.hd95_p95.mean, 1), f(d.hd95_p95.mean, 1), " px") +
    tile("Failed images (IoU < 0.65)", f(u.below.mean, 0), f(d.below.mean, 0)) +
    (sp ? tile("Time per image", f(sp["U-Net"].infer_ms, 1), f(sp["DeepLabV3+"].infer_ms, 1), " ms") : "");

  if (s.by_size) $("#chart-size").innerHTML = dotChart(
    Object.entries(s.by_size).map(([k, v]) => ({ label: `${k}<tspan fill="#9ca3af"> n=${v.n}</tspan>`, u: v["U-Net"], d: v["DeepLabV3+"] })), 0.86, 0.95, 0.02);
  if (s.runs) $("#chart-seeds").innerHTML = seedChart(s.runs);

  const small = s.by_size && s.by_size["< 5%"];
  const items = [
    ["Same overall accuracy", `Dice ${f(u.dice.mean)} vs ${f(d.dice.mean)}: the gap is as small as the spread between training runs.`],
    ["DeepLabV3+: fewer large border errors", `HD95 95th percentile ${f(d.hd95_p95.mean, 1)} vs ${f(u.hd95_p95.mean, 1)} px.`],
    ["U-Net: small lesions and speed", (small ? `Dice ${f(small["U-Net"].mean)} vs ${f(small["DeepLabV3+"].mean)} on lesions under 5% of the image` : "Better on small lesions") +
      (sp ? `, ${(sp["DeepLabV3+"].infer_ms / sp["U-Net"].infer_ms).toFixed(1)}× faster.` : ".")],
  ];
  $("#takeaways").innerHTML = items.map(([t, x]) => `<li><b>${t}</b>${x}</li>`).join("");
}

function renderFailures() {
  const cases = [
    ["ISIC_0013966", "Fluid bubble edge", "The rim of an immersion-fluid bubble is the strongest contour, so both models outline the bubble."],
    ["ISIC_0013567", "Faint, unpigmented lesion", "Both models pick the darker spot next to the pale melanoma in situ."],
    ["ISIC_0011117", "Ambiguous annotation", "The expert mask includes a pale halo; the models follow the visible pigment."],
    ["ISIC_0012160", "Small, diffuse border", "Both over-segment; DeepLabV3+'s coarse 1/4-resolution output is worse here."],
  ];
  const ids = new Set(state.info.examples.map((e) => e.id));
  $("#fails").innerHTML = cases.filter(([id]) => ids.has(id)).map(([id, t, x]) =>
    `<button class="fail" data-id="${id}"><img src="/demo/examples/${id}.jpg" alt=""><b>${t}</b><span>${x}</span></button>`).join("");
  $$(".fail").forEach((b) => b.addEventListener("click", () => run({ example: b.dataset.id })));
}

function dotChart(groups, lo, hi, step) {
  // mean ± std over seeds for both models per category
  const W = 460, H = 220, L = 44, B = 40, T = 10, R = 8;
  const y = (v) => T + (hi - v) / (hi - lo) * (H - T - B);
  const cw = (W - L - R) / groups.length;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
  for (let t = lo; t <= hi + 1e-9; t += step) {
    s += `<line x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}" stroke="#e5e7eb"/><text x="${L - 6}" y="${y(t) + 4}" font-size="11" text-anchor="end" fill="#6b7280">${t.toFixed(2)}</text>`;
  }
  groups.forEach((g, i) => {
    const cx = L + cw * (i + 0.5);
    [["U-Net", g.u, -9], ["DeepLabV3+", g.d, 9]].forEach(([m, v, off]) => {
      if (!v) return;
      const c = css(RGB[m]);
      s += `<line x1="${cx + off}" x2="${cx + off}" y1="${y(v.mean + v.std)}" y2="${y(v.mean - v.std)}" stroke="${c}" stroke-width="2"/>`;
      s += `<circle cx="${cx + off}" cy="${y(v.mean)}" r="5.5" fill="${c}"><title>${m}: ${v.mean.toFixed(3)} ± ${v.std.toFixed(3)}</title></circle>`;
    });
    s += `<text x="${cx}" y="${H - 16}" font-size="12" text-anchor="middle" fill="#15181e">${g.label}</text>`;
  });
  s += `<text x="${L}" y="${H - 1}" font-size="11" fill="#9ca3af">lesion area, % of image · dot = mean, line = ± std over 3 seeds</text>`;
  return s + "</svg>";
}

function seedChart(runs) {
  const vals = runs.map((r) => r.dice);
  const lo = Math.floor((Math.min(...vals) - 0.002) * 500) / 500, hi = Math.ceil((Math.max(...vals) + 0.002) * 500) / 500;
  const W = 460, H = 220, L = 96, R = 14;
  const x = (v) => L + (v - lo) / (hi - lo) * (W - L - R);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
  for (let t = lo; t <= hi + 1e-9; t += 0.002) {
    s += `<line x1="${x(t)}" x2="${x(t)}" y1="20" y2="${H - 44}" stroke="#e5e7eb"/><text x="${x(t)}" y="${H - 28}" font-size="11" text-anchor="middle" fill="#6b7280">${t.toFixed(3)}</text>`;
  }
  MODELS.forEach((m, i) => {
    const yy = 62 + i * 70, rr = runs.filter((r) => r.model === m), c = css(RGB[m]);
    const mu = rr.reduce((a, r) => a + r.dice, 0) / rr.length;
    s += `<text x="0" y="${yy + 4}" font-size="13" font-weight="600" fill="${c}">${m}</text>`;
    s += `<line x1="${x(mu)}" x2="${x(mu)}" y1="${yy - 16}" y2="${yy + 16}" stroke="${c}" stroke-width="2.5"/>`;
    rr.forEach((r) => {
      s += `<circle cx="${x(r.dice)}" cy="${yy}" r="7" fill="${c}" fill-opacity=".75" stroke="#fff" stroke-width="1.5"><title>seed ${r.seed}: ${r.dice.toFixed(4)}</title></circle>`;
    });
  });
  s += `<text x="${L}" y="${H - 6}" font-size="11" fill="#9ca3af">validation Dice · dot = one training run, bar = mean</text>`;
  return s + "</svg>";
}

boot();
