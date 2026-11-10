"""Live demo: U-Net vs DeepLabV3+ (ResNet-34) for skin lesion segmentation, ISIC 2018 Task 1.

    python demo/app.py                 # opens http://127.0.0.1:7860 in the browser
    python demo/app.py --port 8000 --no-browser

Runs fully offline: a small local web server (Python standard library only, no extra packages)
serves one page from demo/static and loads the trained checkpoints from checkpoints/.
The pipeline is the same as in evaluate.py: resize to 256x256 (INTER_AREA), ImageNet
normalisation, model, sigmoid, threshold. Metrics are computed at 256x256 with src/metrics.py,
so for the example images they match results/*/per_image_val.csv.
"""
import argparse
import base64
import hashlib
import json
import sys
import threading
import time
import webbrowser
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.metrics import all_metrics  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import get_device, sync  # noqa: E402

DEMO = ROOT / "demo"
STATIC = DEMO / "static"
EXAMPLES = DEMO / "examples"
SIZE = 256
DISPLAY = 640
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

CHECKPOINTS = {  # (model, seed) -> checkpoint written by train.py with the configs in configs/
    ("U-Net", 42): "checkpoints/unet_r34_best.pt",
    ("U-Net", 1): "checkpoints/unet_r34_seed1_best.pt",
    ("U-Net", 2): "checkpoints/unet_r34_seed2_best.pt",
    ("DeepLabV3+", 42): "checkpoints/deeplabv3plus_r34_best.pt",
    ("DeepLabV3+", 1): "checkpoints/deeplabv3plus_r34_seed1_best.pt",
    ("DeepLabV3+", 2): "checkpoints/deeplabv3plus_r34_seed2_best.pt",
}
RESULT_DIRS = {
    ("U-Net", 42): "results", ("U-Net", 1): "results/unet_r34_seed1", ("U-Net", 2): "results/unet_r34_seed2",
    ("DeepLabV3+", 42): "results/deeplabv3plus_r34", ("DeepLabV3+", 1): "results/deeplabv3plus_r34_seed1",
    ("DeepLabV3+", 2): "results/deeplabv3plus_r34_seed2",
}
MODELS = ("U-Net", "DeepLabV3+")
# figures served to the page (whitelist, relative to the repository root)
FIGURE_DIRS = ("results/comparison", "results/failures", "results/figures", "reports/figures", "results/ablation")


# ----------------------------------------------------------------------------- models

class ModelPool:
    """Loads checkpoints lazily and keeps them in memory; one lock, so requests run one at a time."""

    def __init__(self, device):
        self.device = device
        self.models = {}
        self.lock = threading.Lock()

    def available(self):
        return {m: sorted(s for (mm, s), p in CHECKPOINTS.items() if mm == m and (ROOT / p).exists())
                for m in MODELS}

    def get(self, name, seed):
        key = (name, seed)
        if key not in self.models:
            ckpt = torch.load(ROOT / CHECKPOINTS[key], map_location="cpu", weights_only=False)
            cfg = ckpt["config"]
            model = build_model({**cfg, "model": {**cfg["model"], "encoder_weights": None}})
            model.load_state_dict(ckpt["model"])
            model.to(self.device).eval()
            with torch.no_grad():  # warm-up so the first timed call is honest
                model(torch.zeros(1, 3, SIZE, SIZE, device=self.device))
            sync(self.device)
            self.models[key] = {"model": model, "epoch": ckpt.get("epoch"), "val_dice": ckpt.get("val_dice"),
                                "params_m": sum(p.numel() for p in model.parameters()) / 1e6}
        return self.models[key]

    @torch.no_grad()
    def predict(self, name, seed, image256_rgb):
        entry = self.get(name, seed)
        x = (image256_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1).copy()).float()[None].to(self.device)
        sync(self.device)
        t0 = time.perf_counter()
        prob = torch.sigmoid(entry["model"](x))
        sync(self.device)
        ms = 1000 * (time.perf_counter() - t0)
        return prob.cpu().numpy()[0, 0], ms


# ----------------------------------------------------------------------------- helpers

def png_data_url(arr):
    ok, buf = cv2.imencode(".png", arr)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def jpg_data_url(bgr):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def display_size(w, h):
    s = min(1.0, DISPLAY / max(h, w))
    return max(1, round(w * s)), max(1, round(h * s))


def metrics_at(prob, gt, threshold):
    pred = prob > threshold
    out = {"area_pct": 100 * float(pred.mean()), "empty": bool(not pred.any())}
    if gt is not None:
        out.update({k: float(v) for k, v in all_metrics(pred, gt).items()})
    return out


def lesion_stats(df):
    return {"dice": float(df.dice.mean()), "iou": float(df.iou.mean()),
            "isic": float(np.where(df.iou < 0.65, 0.0, df.iou).mean()),
            "below": int((df.iou < 0.65).sum()), "hd95_med": float(df.hd95.median()),
            "hd95_p95": float(np.percentile(df.hd95, 95)), "n": int(len(df))}


def mean_std(values):
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def build_summary():
    """Everything the 'Results' section shows, read from the committed results/ files."""
    out = {}
    p = ROOT / "results" / "comparison" / "seeds_summary.json"
    if p.exists():
        s = json.load(open(p))
        out["runs"] = s["runs"]
        out["paired"] = s.get("paired_seed_avg")
        agg = {}
        for m in MODELS:
            rr = [r for r in s["runs"] if r["model"] == m]
            agg[m] = {k: mean_std([r[k] for r in rr]) for k in
                      ("dice", "iou", "isic", "below", "hd95_med", "hd95_p95", "train_min", "epochs_run")}
            agg[m]["seeds"] = sorted(r["seed"] for r in rr)
        out["aggregate"] = agg

    p = ROOT / "results" / "comparison" / "speed_benchmark.json"
    if p.exists():
        sp = json.load(open(p))
        out["speed"] = {"chip": sp.get("chip"), **{m: {"epoch_s": v["epoch_s_mean"], "infer_ms": v["inference_ms_mean"]}
                                                    for m, v in sp["models"].items()}}
    p = ROOT / "results" / "comparison" / "comparison.json"
    if p.exists():
        c = json.load(open(p))
        out["params_m"] = {r["model"]: r.get("params_M") for r in c.get("summary", []) if "model" in r}

    # Dice by lesion size and by diagnosis, mean over the three seeds
    meta_p = ROOT / "results" / "comparison" / "per_image_both.csv"
    if meta_p.exists():
        meta = pd.read_csv(meta_p).set_index("isic_id")[["size", "dx"]]
        groups = {"size": ["< 5%", "5-20%", "20-50%", "> 50%"],
                  "dx": ["Melanoma (all)", "Nevus", "Seborrheic keratosis"]}
        tables = {g: {v: {"n": int((meta[g] == v).sum())} for v in vals} for g, vals in groups.items()}
        for m in MODELS:
            per_seed = {g: {v: [] for v in vals} for g, vals in groups.items()}
            for (mm, seed), d in RESULT_DIRS.items():
                f = ROOT / d / "per_image_val.csv"
                if mm != m or not f.exists():
                    continue
                df = pd.read_csv(f).set_index("isic_id").join(meta)
                for g, vals in groups.items():
                    for v in vals:
                        per_seed[g][v].append(df.loc[df[g] == v, "dice"].mean())
            for g, vals in groups.items():
                for v in vals:
                    tables[g][v][m] = mean_std(per_seed[g][v]) if per_seed[g][v] else None
        out["by_size"], out["by_dx"] = tables["size"], tables["dx"]

    # optional ablation results (results/ablation/*.json), shown as-is if present
    abl = []
    for f in sorted((ROOT / "results" / "ablation").glob("*.json")) if (ROOT / "results" / "ablation").is_dir() else []:
        try:
            abl.append({"name": f.stem, "data": json.load(open(f))})
        except json.JSONDecodeError:
            pass
    out["ablation"] = abl

    figs = []
    wanted = [("results/comparison/seeds_dice.png", "Validation Dice of every run (3 seeds per model)"),
              ("results/comparison/learning_curves.png", "Learning curves, U-Net vs DeepLabV3+"),
              ("results/failures/failure_cases.png", "Failure cases that fail in all six runs"),
              ("results/figures/augmentations.png", "Training augmentations"),
              ("results/figures/duplicate_detection.png", "Leakage-aware split: near-duplicate detection")]
    for path, caption in wanted:
        if (ROOT / path).exists():
            figs.append({"src": "/" + path, "caption": caption})
    if (ROOT / "results" / "ablation").is_dir():
        for f in sorted((ROOT / "results" / "ablation").glob("*.png")):
            figs.append({"src": "/" + f.relative_to(ROOT).as_posix(), "caption": "Ablation: " + f.stem.replace("_", " ")})
    out["figures"] = figs
    return out


# ----------------------------------------------------------------------------- app state

class Demo:
    def __init__(self, device):
        self.pool = ModelPool(device)
        self.device = device
        self.examples = json.load(open(EXAMPLES / "examples.json")) if (EXAMPLES / "examples.json").exists() else []
        self.example_ids = {e["id"] for e in self.examples}
        # the files in demo/test_images are byte-identical copies of demo/examples/<id>.jpg, so an uploaded
        # test image is recognised and gets its ground truth (and therefore Dice / IoU / HD95)
        self.hash_to_id = {hashlib.sha1((EXAMPLES / f"{i}.jpg").read_bytes()).hexdigest(): i
                           for i in self.example_ids if (EXAMPLES / f"{i}.jpg").exists()}
        self.cache = OrderedDict()  # key -> {"probs": {...}, "gt": ...}
        self.summary = build_summary()

    def info(self):
        avail = self.pool.available()
        default = {m: (42 if 42 in s else (s[0] if s else None)) for m, s in avail.items()}
        return {"device": str(self.device), "available": avail, "default_seeds": default,
                "examples": self.examples, "summary": self.summary}

    def _load_input(self, req):
        """Returns (key, image256 RGB, gt256 bool or None, display BGR or None, gt_display or None)."""
        if req.get("example"):
            ex = req["example"]
            if ex not in self.example_ids:
                raise ValueError("unknown example")
            bgr = cv2.imread(str(EXAMPLES / f"{ex}_256.png"), cv2.IMREAD_COLOR)
            gt = cv2.imread(str(EXAMPLES / f"{ex}_gt256.png"), cv2.IMREAD_GRAYSCALE) > 127
            disp = cv2.imread(str(EXAMPLES / f"{ex}.jpg"), cv2.IMREAD_COLOR)
            return ex, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), gt, disp, None
        data = req.get("image", "")
        raw = base64.b64decode(data.split(",", 1)[-1])
        if len(raw) > MAX_UPLOAD_BYTES:
            raise ValueError("image is larger than 25 MB")
        known = self.hash_to_id.get(hashlib.sha1(raw).hexdigest())
        if known:
            return self._load_input({"example": known})
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("could not read the image (use JPG or PNG)")
        h, w = img.shape[:2]
        disp = cv2.resize(img, display_size(w, h), interpolation=cv2.INTER_AREA)
        small = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        return "up_" + hashlib.sha1(raw).hexdigest()[:16], cv2.cvtColor(small, cv2.COLOR_BGR2RGB), None, disp, True

    def predict(self, req):
        threshold = float(req.get("threshold", 0.5))
        key, img256, gt, disp, uploaded = self._load_input(req)
        seeds = req.get("seeds", {})
        avail = self.pool.available()
        dh, dw = disp.shape[:2]
        result = {"key": key, "width": dw, "height": dh, "threshold": threshold, "models": {},
                  "example": key if key in self.example_ids else None}
        if uploaded:
            result["image"] = jpg_data_url(disp)
        else:
            result["image"] = f"/demo/examples/{key}.jpg"
            result["gt"] = f"/demo/examples/{key}_gt.png"

        probs = {}
        with self.pool.lock:
            for m in MODELS:
                seed = int(seeds.get(m) or 0)
                if seed not in avail[m]:
                    seed = 42 if 42 in avail[m] else (avail[m][0] if avail[m] else None)
                if seed is None:
                    continue
                prob, ms = self.pool.predict(m, seed, img256)
                probs[m] = prob
                info = self.pool.models[(m, seed)]
                prob_disp = cv2.resize(prob, (dw, dh), interpolation=cv2.INTER_LINEAR)
                result["models"][m] = {
                    "seed": seed, "ms": ms, "epoch": info["epoch"], "params_m": info["params_m"],
                    "prob": png_data_url(np.clip(prob_disp * 255 + 0.5, 0, 255).astype(np.uint8)),
                    "metrics": metrics_at(prob, gt, threshold),
                }
        self.cache[key] = {"probs": probs, "gt": gt}
        self.cache.move_to_end(key)
        while len(self.cache) > 64:
            self.cache.popitem(last=False)
        result["agreement"] = self.agreement(probs, threshold)
        return result

    @staticmethod
    def agreement(probs, threshold):
        if len(probs) < 2:
            return None
        a, b = (probs[m] > threshold for m in MODELS)
        return {k: float(v) for k, v in all_metrics(a, b).items()}

    def rescore(self, req):
        threshold = float(req.get("threshold", 0.5))
        c = self.cache.get(req.get("key"))
        if c is None:
            raise ValueError("expired, run the prediction again")
        return {"threshold": threshold,
                "models": {m: {"metrics": metrics_at(p, c["gt"], threshold)} for m, p in c["probs"].items()},
                "agreement": self.agreement(c["probs"], threshold)}


# ----------------------------------------------------------------------------- HTTP

CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".png": "image/png", ".jpg": "image/jpeg",
                 ".svg": "image/svg+xml", ".json": "application/json"}


def make_handler(demo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the terminal quiet during the presentation
            pass

        def _send(self, status, body, ctype):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=HTTPStatus.OK):
            self._send(status, json.dumps(obj).encode(), "application/json")

        def _file(self, path):
            if not path.is_file():
                return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            self._send(HTTPStatus.OK, path.read_bytes(), CONTENT_TYPES.get(path.suffix, "application/octet-stream"))

        def do_GET(self):
            url = self.path.split("?", 1)[0]
            if url in ("/", "/index.html"):
                return self._file(STATIC / "index.html")
            if url == "/api/info":
                return self._json(demo.info())
            rel = url.lstrip("/")
            target = (ROOT / rel).resolve()
            allowed = [STATIC, EXAMPLES] + [ROOT / d for d in FIGURE_DIRS]
            if rel.startswith("static/"):
                target = (DEMO / rel).resolve()
            if any(target.is_relative_to(a.resolve()) for a in allowed):
                return self._file(target)
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > MAX_UPLOAD_BYTES * 1.4:
                    raise ValueError("image is larger than 25 MB")
                req = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/api/predict":
                    return self._json(demo.predict(req))
                if self.path == "/api/rescore":
                    return self._json(demo.rescore(req))
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except ValueError as e:
                self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            except Exception as e:  # show the error in the page instead of a silent failure
                self._json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--device", default="mps", help="mps, cuda or cpu (falls back to cpu)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    demo = Demo(get_device(args.device))
    avail = demo.pool.available()
    print(f"device: {demo.device}")
    for m in MODELS:
        print(f"{m:11s} checkpoints for seeds: {avail[m] or 'none found in checkpoints/'}")
    if not any(avail.values()):
        print("No checkpoints found. Train the models (README) or copy the *_best.pt files into checkpoints/.")
    print("loading default models ...")
    for m, s in demo.info()["default_seeds"].items():
        if s is not None:
            demo.pool.get(m, s)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(demo))
    url = f"http://{args.host}:{args.port}"
    print(f"demo running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
