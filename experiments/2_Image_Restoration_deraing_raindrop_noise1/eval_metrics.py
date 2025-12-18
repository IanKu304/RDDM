# eval_metrics.py
# Compute PSNR/SSIM (+ optional LPIPS/FID) between restored outputs (pred) and GT on a fixed test set.
# Windows-friendly; supports matching by basename or by order.
#
# Example:
#   python eval_metrics.py ^
#     --gt_flist ".\data_gtrain_local\test_gt.flist" ^
#     --pred_dir ".\results\RUN_NAME\test_timestep_10\ckpt_60" ^
#     --match basename --resize pred_to_gt ^
#     --lpips --fid --device cuda ^
#     --out_dir ".\results\RUN_NAME\metrics"

import argparse
import os
import sys
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def read_flist(path: Path) -> list[str]:
    lines = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            lines.append(s)
    return lines


def list_images_recursive(root: Path) -> list[Path]:
    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return paths


def build_pred_index(pred_dir: Path) -> dict[str, Path]:
    idx = {}
    for p in list_images_recursive(pred_dir):
        key = p.name  # basename with extension
        if key in idx:
            raise RuntimeError(f"Duplicate basename in pred_dir: {key}\n  {idx[key]}\n  {p}")
        idx[key] = p
    return idx


def pil_to_np_rgb01(img: Image.Image) -> np.ndarray:
    # RGB float32 in [0,1]
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def resize_to(img: Image.Image, size_wh: tuple[int, int]) -> Image.Image:
    # size_wh = (W,H)
    return img.resize(size_wh, resample=Image.BICUBIC)


def compute_psnr(gt01: np.ndarray, pred01: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(gt01, pred01, data_range=1.0))


def compute_ssim(gt01: np.ndarray, pred01: np.ndarray) -> float:
    # channel_axis for new skimage
    return float(structural_similarity(gt01, pred01, data_range=1.0, channel_axis=-1))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser("RDDM metrics: PSNR/SSIM (+LPIPS/FID optional)")

    # Required
    ap.add_argument("--gt_flist", type=str, required=True, help="Path to test_gt.flist (one GT path per line)")
    ap.add_argument("--pred_dir", type=str, required=True, help="Directory containing restored output images")

    # Matching
    ap.add_argument("--match", type=str, default="basename", choices=["basename", "order"],
                    help="How to align pred with GT: basename (recommended) or order")
    ap.add_argument("--allow_missing", action="store_true", help="Skip missing pairs instead of failing")

    # Resize policy
    ap.add_argument("--resize", type=str, default="pred_to_gt",
                    choices=["none", "pred_to_gt", "gt_to_pred"],
                    help="If sizes differ, resize one side to match the other")

    # Optional metrics
    ap.add_argument("--lpips", action="store_true", help="Compute LPIPS (perceptual)")
    ap.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    ap.add_argument("--fid", action="store_true", help="Compute FID(pred vs GT) on matched test set")

    # Device for lpips/fid (fid can run on cpu too)
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda")

    # Output
    ap.add_argument("--out_dir", type=str, default="", help="Where to write per_image.csv, summary.json")
    ap.add_argument("--gt_eval_dir", type=str, default="",
                    help="Optional: existing GT directory for FID (if not provided, will build from flist)")

    args = ap.parse_args()

    gt_flist = Path(args.gt_flist).resolve()
    pred_dir = Path(args.pred_dir).resolve()

    if not gt_flist.exists():
        raise FileNotFoundError(f"gt_flist not found: {gt_flist}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"pred_dir not found: {pred_dir}")

    # out_dir default
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (pred_dir / "metrics")
    ensure_dir(out_dir)

    gt_paths = read_flist(gt_flist)
    if len(gt_paths) == 0:
        raise RuntimeError(f"Empty flist: {gt_flist}")

    # Resolve GT paths (keep as-is if absolute; otherwise relative to flist parent)
    gt_resolved: list[Path] = []
    for s in gt_paths:
        p = Path(s)
        if not p.is_absolute():
            p = (gt_flist.parent / p).resolve()
        else:
            p = p.resolve()
        gt_resolved.append(p)

    # Pairing
    pairs = []
    missing = []

    if args.match == "basename":
        pred_index = build_pred_index(pred_dir)
        for gt in gt_resolved:
            key = gt.name  # basename with ext
            pred = pred_index.get(key, None)
            if pred is None:
                missing.append(str(gt))
                continue
            pairs.append((gt, pred))
    else:
        pred_imgs = sorted(list_images_recursive(pred_dir))
        if len(pred_imgs) != len(gt_resolved) and not args.allow_missing:
            raise RuntimeError(
                f"Order matching requires same counts.\nGT={len(gt_resolved)} pred={len(pred_imgs)}.\n"
                f"Use --allow_missing or switch to --match basename."
            )
        n = min(len(pred_imgs), len(gt_resolved))
        for i in range(n):
            pairs.append((gt_resolved[i], pred_imgs[i]))

    if missing and not args.allow_missing:
        msg = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} GT basenames in pred_dir.\n"
            f"First 10 missing GT paths:\n{msg}\n\n"
            f"Fix: ensure pred output filenames match GT basenames, or use --match order, or regenerate outputs."
        )

    if len(pairs) == 0:
        raise RuntimeError("No matched pairs found. Check --pred_dir and --gt_flist / --match.")

    # Optional LPIPS init
    lpips_model = None
    torch = None
    if args.lpips:
        import torch as _torch
        import lpips as _lpips
        torch = _torch
        device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
        lpips_model = _lpips.LPIPS(net=args.lpips_net).to(device)
        lpips_model.eval()

    rows = []
    for gt_path, pred_path in tqdm(pairs, desc="Computing PSNR/SSIM/LPIPS"):
        if not gt_path.exists() or not pred_path.exists():
            if args.allow_missing:
                continue
            raise FileNotFoundError(f"Missing file:\n  gt={gt_path}\n  pred={pred_path}")

        gt_img = Image.open(gt_path)
        pred_img = Image.open(pred_path)

        # Resize if needed
        if gt_img.size != pred_img.size:
            if args.resize == "pred_to_gt":
                pred_img = resize_to(pred_img, gt_img.size)
            elif args.resize == "gt_to_pred":
                gt_img = resize_to(gt_img, pred_img.size)
            else:
                raise RuntimeError(f"Size mismatch gt={gt_img.size} pred={pred_img.size}. Use --resize pred_to_gt")

        gt01 = pil_to_np_rgb01(gt_img)
        pred01 = pil_to_np_rgb01(pred_img)

        psnr = compute_psnr(gt01, pred01)
        ssim = compute_ssim(gt01, pred01)

        lp = None
        if lpips_model is not None:
            # LPIPS expects tensor in [-1, 1], shape (1,3,H,W)
            device = next(lpips_model.parameters()).device
            gt_t = torch.from_numpy(gt01).permute(2, 0, 1).unsqueeze(0).to(device)
            pr_t = torch.from_numpy(pred01).permute(2, 0, 1).unsqueeze(0).to(device)
            gt_t = gt_t * 2.0 - 1.0
            pr_t = pr_t * 2.0 - 1.0
            with torch.no_grad():
                lp = float(lpips_model(pr_t, gt_t).mean().item())

        rows.append({
            "gt_path": str(gt_path),
            "pred_path": str(pred_path),
            "name": gt_path.name,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lp
        })

    df = pd.DataFrame(rows)
    per_image_csv = out_dir / "per_image.csv"
    df.to_csv(per_image_csv, index=False, encoding="utf-8-sig")

    summary = {
        "count": int(len(df)),
        "psnr_mean": float(df["psnr"].mean()),
        "psnr_std": float(df["psnr"].std(ddof=1)) if len(df) > 1 else 0.0,
        "ssim_mean": float(df["ssim"].mean()),
        "ssim_std": float(df["ssim"].std(ddof=1)) if len(df) > 1 else 0.0,
    }
    if args.lpips:
        summary["lpips_mean"] = float(df["lpips"].dropna().mean())
        summary["lpips_std"] = float(df["lpips"].dropna().std(ddof=1)) if df["lpips"].notna().sum() > 1 else 0.0

    # Optional FID
    fid_value = None
    if args.fid:
        try:
            from cleanfid import fid as clean_fid
        except Exception as e:
            print("[WARN] clean-fid not available. Install with: pip install clean-fid")
            print("       Error:", repr(e))
            clean_fid = None

        if clean_fid is not None:
            # Build GT eval dir containing ONLY matched GT images (for fixed test set)
            if args.gt_eval_dir:
                gt_eval_dir = Path(args.gt_eval_dir).resolve()
                if not gt_eval_dir.exists():
                    raise FileNotFoundError(f"--gt_eval_dir not found: {gt_eval_dir}")
            else:
                gt_eval_dir = out_dir / "_gt_eval"
                if gt_eval_dir.exists():
                    shutil.rmtree(gt_eval_dir)
                ensure_dir(gt_eval_dir)

                # Copy matched GTs (basename preserved)
                for gt_path, _ in pairs:
                    if gt_path.exists():
                        safe_copy(gt_path, gt_eval_dir / gt_path.name)

            # NOTE: FID is distribution-level: compares pred_dir vs gt_eval_dir (fixed test set)
            # device: "cuda" or "cpu"
            device = args.device
            if device != "cpu":
                # clean-fid accepts "cuda" (uses torch)
                device = "cuda"

            fid_value = float(clean_fid.compute_fid(
                str(pred_dir),
                str(gt_eval_dir),
                mode="clean",
                device=device
            ))
            summary["fid_pred_vs_gt"] = fid_value
            summary["fid_note"] = "FID(pred_outputs vs GT) on the fixed matched test set distribution"

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved:\n  {per_image_csv}\n  {summary_json}")


if __name__ == "__main__":
    main()
