import os
import sys
import csv
import json
import argparse
import subprocess
from pathlib import Path

from src.denoising_diffusion_pytorch import GaussianDiffusion
from src.residual_denoising_diffusion_pytorch import (
    ResidualDiffusion,
    Trainer,
    Unet,
    UnetRes,
    set_seed,
)

def _git_commit_short() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

def main() -> None:
    parser = argparse.ArgumentParser("RDDM GT-RAIN test + metrics (Windows)")
    parser.add_argument("--run_dir", type=str, required=True, help="例如 results/<run_name>")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--sampling_timesteps", type=int, default=10)

    parser.add_argument("--ckpt", type=str, default="latest", help="'latest' 或 milestone 整數（例如 80）")
    parser.add_argument("--eval_csv", type=str, default="ablation_results.csv", help="把 summary append 到這個 CSV")
    parser.add_argument("--gt_flist", type=str, default="", help="預設用 data_gtrain/test_gt.flist")

    parser.add_argument("--compute_fid", action="store_true")
    parser.add_argument("--compute_lpips", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.stdout.flush()
    set_seed(args.seed)

    HERE = Path(__file__).resolve().parent

    folder = [
        str(HERE / "data_gtrain" / "train_gt.flist"),
        str(HERE / "data_gtrain" / "train_input.flist"),
        str(HERE / "data_gtrain" / "test_gt.flist"),
        str(HERE / "data_gtrain" / "test_input.flist"),
    ]

    # 保持跟 train 一樣的 model 設定（restoration）
    original_ddim_ddpm = False
    condition = True
    input_condition = False
    input_condition_mask = False

    image_size = 256
    train_num_steps = 120000
    save_and_sample_every = 1000

    if original_ddim_ddpm:
        model = Unet(dim=64, dim_mults=(1, 2, 4, 8))
        diffusion = GaussianDiffusion(
            model,
            image_size=image_size,
            timesteps=1000,
            sampling_timesteps=250,
            loss_type="l1",
        )
    else:
        model = UnetRes(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            share_encoder=0,
            condition=condition,
            input_condition=input_condition,
        )
        diffusion = ResidualDiffusion(
            model,
            image_size=image_size,
            timesteps=1000,
            sampling_timesteps=args.sampling_timesteps,
            objective="pred_res_noise",
            loss_type="l1",
            condition=condition,
            sum_scale=1,
            input_condition=input_condition,
            input_condition_mask=input_condition_mask,
        )

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    trainer = Trainer(
        diffusion,
        folder,
        train_batch_size=1,
        num_samples=1,
        train_lr=8e-5,
        train_num_steps=train_num_steps,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=False,
        convert_image_to="RGB",
        condition=condition,
        save_and_sample_every=save_and_sample_every,
        equalizeHist=False,
        crop_patch=False,
        generation=False,
    )

    # ckpt 選擇
    if args.ckpt == "latest":
        import re
        ms = None
        for ckpt in run_dir.glob("model-*.pt"):
            m = re.search(r"model-(\d+)\.pt$", ckpt.name)
            if m:
                v = int(m.group(1))
                ms = v if ms is None else max(ms, v)
        if ms is None:
            raise RuntimeError("No checkpoint found under run_dir.")
    else:
        ms = int(args.ckpt)

    trainer.load(ms)

    pred_dir = run_dir / f"test_timestep_{args.sampling_timesteps}" / f"ckpt_{ms}"
    pred_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(trainer, "set_results_folder"):
        trainer.set_results_folder(str(pred_dir))

    # 產生復原結果（圖片輸出位置看你們 Trainer.test 實作）
    trainer.test(last=True)

    # metrics
    gt_flist = args.gt_flist.strip()
    if not gt_flist:
        gt_flist = str(HERE / "data_gtrain" / "test_gt.flist")

    eval_script = HERE / "eval_metrics.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"Missing eval_metrics.py at: {eval_script}")

    cmd = [
        sys.executable, str(eval_script),
        "--pred_dir", str(pred_dir),
        "--gt_flist", gt_flist,
        "--out_json", str(pred_dir / "metrics.json"),
    ]
    if args.compute_fid:
        cmd.append("--compute_fid")
    if args.compute_lpips:
        cmd.append("--compute_lpips")

    print("[INFO] Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    metrics = json.loads((pred_dir / "metrics.json").read_text(encoding="utf-8"))

    # append CSV
    out_csv = Path(args.eval_csv)
    header = [
        "run_dir", "ckpt", "sampling_timesteps",
        "psnr_mean", "ssim_mean",
        "lpips_mean", "fid",
        "git_commit", "gpu", "torch_version",
    ]
    row = [
        str(run_dir),
        ms,
        args.sampling_timesteps,
        metrics.get("psnr_mean"),
        metrics.get("ssim_mean"),
        metrics.get("lpips_mean"),
        metrics.get("fid"),
        _git_commit_short(),
        metrics.get("gpu"),
        metrics.get("torch_version"),
    ]

    write_header = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

    print(f"[INFO] Appended summary to: {out_csv}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
