"""
compare_pt_onnx.py — parity and latency check: YOLOv8 .pt vs its ONNX export
==============================================================================

Runs BOTH models over every image in a folder, with identical settings, and
reports two things:

  1. PARITY  — do they find the same objects? Detections are matched by class
               and box overlap; confidence and box differences are measured.
  2. LATENCY — how long does predict() take for each (median of several runs)?

Images are loaded with cv2.imread, which returns BGR. Ultralytics expects BGR
for numpy input, so they are passed to both models as-is (no RGB conversion).

Usage:
    python compare_pt_onnx.py --images my_photos --pt models/best.pt --onnx models/best.onnx

Outputs (in --out, default parity_out/):
    parity_summary.csv    one row per image
    parity_details.json   every detection from both models + environment info
    report.txt            the same summary that is printed to the console

Notes:
  - Tolerances below are starting points, not standards. Small numeric
    differences between PyTorch and ONNX Runtime are normal.
  - Run it in an environment with the same Ultralytics version as your Docker
    image (8.0.222) if you can; the versions used are recorded in the report.
  - Latency on a shared or busy machine is noisy. Close other programs and
    repeat the whole script 2-3 times before quoting any number.
"""

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ── Tolerances (see note above) ─────────────────────────────────────────
TOL_CONF = 0.02        # max confidence difference counted as "same"
TOL_BOX_PX = 2.0       # max box-coordinate difference (pixels) counted as "same"
HARD_CONF = 0.05       # beyond this the image is a MISMATCH
HARD_BOX_PX = 5.0
BORDERLINE_MARGIN = 0.02   # an unmatched detection this close to the conf
                           # threshold may just be flipping in/out of the cutoff


# ════════════════════════════════════════════════════════
# 1. PURE LOGIC (no model / image libraries needed)
# ════════════════════════════════════════════════════════

def iou_xyxy(a: List[float], b: List[float]) -> float:
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def extract_detections(result: Any) -> List[Dict[str, Any]]:
    """Converts an Ultralytics Results object into plain dicts, highest confidence first."""
    dets = []
    for box in result.boxes:
        cid = int(box.cls[0])
        dets.append({
            "class_id": cid,
            "class_name": str(result.names[cid]),
            "confidence": float(box.conf[0]),
            "bbox": [float(x) for x in box.xyxy[0]],
        })
    dets.sort(key=lambda d: d["confidence"], reverse=True)
    return dets


def match_detections(
    dets_a: List[Dict[str, Any]],
    dets_b: List[Dict[str, Any]],
    match_iou: float = 0.50,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Greedy matching: for each detection in A (highest confidence first), pick the
    unmatched detection in B with the SAME class and the highest IoU >= match_iou.

    match_iou is deliberately loose (0.5): it only PAIRS boxes. How different the
    paired boxes are is judged afterwards by the pixel / confidence tolerances.
    A strict value here would turn a harmless 3 px shift on a small box into
    two "unmatched" detections. Within one model NMS already keeps same-class
    boxes below IoU 0.5, so an IoU above 0.5 across models means the same object.
    Returns (matches as (idx_a, idx_b, iou), unmatched indices in A, unmatched in B).
    """
    used_b = set()
    matches = []
    unmatched_a = []
    for i, a in enumerate(dets_a):
        best_j, best_iou = None, 0.0
        for j, b in enumerate(dets_b):
            if j in used_b or b["class_id"] != a["class_id"]:
                continue
            v = iou_xyxy(a["bbox"], b["bbox"])
            if v >= match_iou and v > best_iou:
                best_j, best_iou = j, v
        if best_j is None:
            unmatched_a.append(i)
        else:
            used_b.add(best_j)
            matches.append((i, best_j, best_iou))
    unmatched_b = [j for j in range(len(dets_b)) if j not in used_b]
    return matches, unmatched_a, unmatched_b


def compare_image(
    dets_pt: List[Dict[str, Any]],
    dets_onnx: List[Dict[str, Any]],
    conf_threshold: float,
    match_iou: float = 0.50,
    tol_conf: float = TOL_CONF,
    tol_box: float = TOL_BOX_PX,
) -> Dict[str, Any]:
    """Parity verdict for one image."""
    matches, un_pt, un_onnx = match_detections(dets_pt, dets_onnx, match_iou)

    max_conf_diff, max_box_diff = 0.0, 0.0
    for i, j, _ in matches:
        max_conf_diff = max(max_conf_diff, abs(dets_pt[i]["confidence"] - dets_onnx[j]["confidence"]))
        box_diff = max(abs(p - q) for p, q in zip(dets_pt[i]["bbox"], dets_onnx[j]["bbox"]))
        max_box_diff = max(max_box_diff, box_diff)

    # Unmatched detections close to the confidence cutoff may just be flipping
    # in/out of it — flag them separately from clear disagreements.
    cutoff = conf_threshold + BORDERLINE_MARGIN
    real_unmatched = (
        [dets_pt[i] for i in un_pt if dets_pt[i]["confidence"] >= cutoff]
        + [dets_onnx[j] for j in un_onnx if dets_onnx[j]["confidence"] >= cutoff]
    )

    if real_unmatched or max_conf_diff > HARD_CONF or max_box_diff > HARD_BOX_PX:
        status = "MISMATCH"
    elif not un_pt and not un_onnx and max_conf_diff <= tol_conf and max_box_diff <= tol_box:
        status = "OK"
    else:
        status = "BORDERLINE"

    return {
        "n_pt": len(dets_pt),
        "n_onnx": len(dets_onnx),
        "matched": len(matches),
        "unmatched_pt": len(un_pt),
        "unmatched_onnx": len(un_onnx),
        "max_conf_diff": round(max_conf_diff, 4),
        "max_box_diff_px": round(max_box_diff, 2),
        "status": status,
        "unmatched_pt_dets": [dets_pt[i] for i in un_pt],
        "unmatched_onnx_dets": [dets_onnx[j] for j in un_onnx],
    }


def class_counts(dets: List[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for d in dets:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
    return "|".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "none"


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


# ════════════════════════════════════════════════════════
# 2. MODEL / IMAGE HELPERS (need cv2 + ultralytics at run time)
# ════════════════════════════════════════════════════════

def load_image(path: Path):
    """Reads an image as a BGR numpy array (the cv2 convention Ultralytics expects)."""
    import cv2
    img = cv2.imread(str(path))
    return img  # None if unreadable


def run_once(model: Any, img: Any, conf: float, iou: float, imgsz: int) -> Tuple[Any, float, float]:
    """One predict() call. Returns (result, wall_ms, inference_only_ms)."""
    t0 = time.perf_counter()
    res = model.predict(source=img, conf=conf, iou=iou, imgsz=imgsz, device="cpu", verbose=False)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    r = res[0]
    infer_ms = float(getattr(r, "speed", {}).get("inference", float("nan")))
    return r, wall_ms, infer_ms


def environment_info(pt_path: Path, onnx_path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    for mod in ("ultralytics", "torch", "onnxruntime", "onnx", "cv2", "numpy"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = "not installed"
    info["pt_size_mb"] = round(os.path.getsize(pt_path) / 1e6, 2) if pt_path.exists() else None
    info["onnx_size_mb"] = round(os.path.getsize(onnx_path) / 1e6, 2) if onnx_path.exists() else None
    return info


# ════════════════════════════════════════════════════════
# 3. THE COMPARISON LOOP
# ════════════════════════════════════════════════════════

def run_comparison(
    model_pt: Any,
    model_onnx: Any,
    images: List[Path],
    conf: float,
    iou: float,
    imgsz: int,
    warmup: int,
    repeats: int,
    match_iou: float,
    tol_conf: float,
    tol_box: float,
    log=print,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []

    # Warmup on the first readable image so first-call overhead is not timed.
    first = next((load_image(p) for p in images if load_image(p) is not None), None)
    if first is None:
        raise RuntimeError("No readable images in the folder.")
    log(f"Warming up both models ({warmup} calls each)...")
    for _ in range(warmup):
        run_once(model_pt, first, conf, iou, imgsz)
        run_once(model_onnx, first, conf, iou, imgsz)

    for n, path in enumerate(images, 1):
        img = load_image(path)
        if img is None:
            log(f"[{n}/{len(images)}] {path.name}: could not be read — skipped")
            continue

        pt_wall, onnx_wall, pt_inf, onnx_inf = [], [], [], []
        res_pt = res_onnx = None
        for rep in range(repeats):
            # Alternate which model goes first to avoid an order bias.
            order = ("pt", "onnx") if rep % 2 == 0 else ("onnx", "pt")
            for which in order:
                if which == "pt":
                    res_pt, w, i_ms = run_once(model_pt, img, conf, iou, imgsz)
                    pt_wall.append(w); pt_inf.append(i_ms)
                else:
                    res_onnx, w, i_ms = run_once(model_onnx, img, conf, iou, imgsz)
                    onnx_wall.append(w); onnx_inf.append(i_ms)

        dets_pt = extract_detections(res_pt)
        dets_onnx = extract_detections(res_onnx)
        cmp = compare_image(dets_pt, dets_onnx, conf, match_iou, tol_conf, tol_box)

        row = {
            "image": path.name,
            "n_pt": cmp["n_pt"], "n_onnx": cmp["n_onnx"], "matched": cmp["matched"],
            "unmatched_pt": cmp["unmatched_pt"], "unmatched_onnx": cmp["unmatched_onnx"],
            "max_conf_diff": cmp["max_conf_diff"], "max_box_diff_px": cmp["max_box_diff_px"],
            "status": cmp["status"],
            "pt_ms_median": round(statistics.median(pt_wall), 2),
            "onnx_ms_median": round(statistics.median(onnx_wall), 2),
            "pt_infer_ms_median": round(statistics.median(pt_inf), 2),
            "onnx_infer_ms_median": round(statistics.median(onnx_inf), 2),
            "classes_pt": class_counts(dets_pt),
            "classes_onnx": class_counts(dets_onnx),
        }
        rows.append(row)
        details.append({
            "image": path.name,
            "pt_detections": dets_pt,
            "onnx_detections": dets_onnx,
            "unmatched_pt": cmp["unmatched_pt_dets"],
            "unmatched_onnx": cmp["unmatched_onnx_dets"],
        })
        log(f"[{n}/{len(images)}] {path.name:<32} pt={cmp['n_pt']:>2} onnx={cmp['n_onnx']:>2} "
            f"| {cmp['status']:<10} | pt {row['pt_ms_median']:>7.1f} ms  onnx {row['onnx_ms_median']:>7.1f} ms")
    return rows, details


def build_report(rows: List[Dict[str, Any]], env: Dict[str, Any], args: Dict[str, Any]) -> str:
    L: List[str] = []
    ok = sum(r["status"] == "OK" for r in rows)
    bd = sum(r["status"] == "BORDERLINE" for r in rows)
    mm = sum(r["status"] == "MISMATCH" for r in rows)
    L.append("=" * 72)
    L.append("PT vs ONNX COMPARISON REPORT")
    L.append("=" * 72)
    L.append(f"Images compared : {len(rows)}")
    L.append(f"Settings        : conf={args['conf']} iou={args['iou']} imgsz={args['imgsz']} "
             f"warmup={args['warmup']} repeats={args['repeats']}")
    L.append(f"Tolerances      : conf diff <= {args['tol_conf']}, box diff <= {args['tol_box']} px "
             f"(match IoU >= {args['match_iou']})")
    L.append("")
    L.append("PARITY")
    L.append(f"  OK         : {ok}")
    L.append(f"  BORDERLINE : {bd}   (small numeric differences, or a detection right at the cutoff)")
    L.append(f"  MISMATCH   : {mm}   (clear disagreement — look at these images)")
    total_pt = sum(r["n_pt"] for r in rows)
    total_onnx = sum(r["n_onnx"] for r in rows)
    L.append(f"  Total detections: pt={total_pt}  onnx={total_onnx}")
    if rows:
        L.append(f"  Largest confidence difference on matched boxes: {max(r['max_conf_diff'] for r in rows)}")
        L.append(f"  Largest box difference on matched boxes      : {max(r['max_box_diff_px'] for r in rows)} px")
    flagged = [r for r in rows if r["status"] != "OK"]
    if flagged:
        L.append("")
        L.append("  Images to look at:")
        for r in flagged:
            L.append(f"    {r['status']:<10} {r['image']:<32} pt={r['classes_pt']}  onnx={r['classes_onnx']}")
    L.append("")
    L.append("LATENCY (wall-clock around predict(): preprocess + inference + NMS, CPU)")
    if rows:
        pt = [r["pt_ms_median"] for r in rows]
        on = [r["onnx_ms_median"] for r in rows]
        L.append(f"  .pt   : median {statistics.median(pt):7.1f} ms | mean {statistics.mean(pt):7.1f} | p90 {percentile(pt, 90):7.1f}")
        L.append(f"  .onnx : median {statistics.median(on):7.1f} ms | mean {statistics.mean(on):7.1f} | p90 {percentile(on, 90):7.1f}")
        ratio = statistics.median(on) / statistics.median(pt) if statistics.median(pt) else float("nan")
        faster = sum(o < p for o, p in zip(on, pt))
        L.append(f"  onnx/pt median ratio: {ratio:.2f}  (below 1.00 means ONNX is faster)")
        L.append(f"  ONNX faster on {faster} of {len(rows)} images")
        L.append("  Do not quote a speedup unless it holds across images AND across repeated runs of this script.")
    L.append("")
    L.append("ENVIRONMENT")
    for k, v in env.items():
        L.append(f"  {k:<14}: {v}")
    L.append("")
    return "\n".join(L)


# ════════════════════════════════════════════════════════
# 4. CLI
# ════════════════════════════════════════════════════════

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compare a YOLOv8 .pt model with its ONNX export.")
    ap.add_argument("--images", required=True, type=Path, help="folder of test images")
    ap.add_argument("--pt", required=True, type=Path, help="path to best.pt")
    ap.add_argument("--onnx", required=True, type=Path, help="path to best.onnx")
    ap.add_argument("--out", type=Path, default=Path("parity_out"))
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--match-iou", type=float, default=0.50,
                    help="min IoU to pair a .pt box with an .onnx box (pairing only; "
                         "box/confidence differences are judged separately)")
    ap.add_argument("--tol-conf", type=float, default=TOL_CONF)
    ap.add_argument("--tol-box", type=float, default=TOL_BOX_PX)
    a = ap.parse_args(argv)

    for p in (a.pt, a.onnx):
        if not p.exists():
            print(f"ERROR: file not found: {p}")
            return 1
    if not a.images.is_dir():
        print(f"ERROR: not a folder: {a.images}")
        return 1
    images = sorted(p for p in a.images.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"ERROR: no images ({', '.join(sorted(IMG_EXTS))}) in {a.images}")
        return 1

    from ultralytics import YOLO
    print(f"Loading {a.pt.name} and {a.onnx.name} ...")
    model_pt = YOLO(str(a.pt))
    model_onnx = YOLO(str(a.onnx), task="detect")

    rows, details = run_comparison(
        model_pt, model_onnx, images, a.conf, a.iou, a.imgsz,
        a.warmup, a.repeats, a.match_iou, a.tol_conf, a.tol_box,
    )
    if not rows:
        print("No images were processed.")
        return 1

    env = environment_info(a.pt, a.onnx)
    args_dict = {"conf": a.conf, "iou": a.iou, "imgsz": a.imgsz, "warmup": a.warmup,
                 "repeats": a.repeats, "match_iou": a.match_iou,
                 "tol_conf": a.tol_conf, "tol_box": a.tol_box}
    report = build_report(rows, env, args_dict)

    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "parity_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(a.out / "parity_details.json", "w", encoding="utf-8") as f:
        json.dump({"environment": env, "settings": args_dict, "images": details}, f, indent=2)
    with open(a.out / "report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + report)
    print(f"Saved: {a.out / 'parity_summary.csv'}, parity_details.json, report.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
