import os
import sys
import json
import platform
import argparse
import subprocess
import inspect
import re
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

def _gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "cpu"
    except Exception:
        return "unknown"

def _torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except Exception:
        return "unknown"

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _find_latest_milestone(run_dir: Path):
    best = None
    for ckpt in run_dir.glob("model-*.pt"):
        m = re.search(r"model-(\d+)\.pt$", ckpt.name)
        if not m:
            continue
        ms = int(m.group(1))
        if (best is None) or (ms > best):
            best = ms
    return best

def _build_run_name(args: argparse.Namespace) -> str:
    def fmt(x: float) -> str:
        s = f"{x:.4f}".rstrip("0").rstrip(".")
        return s
    return (
        f"GT-RAIN__img{args.image_size}__bs{args.batch_size}__acc{args.grad_accum}__"
        f"steps{args.train_num_steps}__sched{args.beta_schedule}__"
        f"betaEnd{fmt(args.beta_end)}__betaScale{fmt(args.beta_scale)}__seed{args.seed}"
    )

def main() -> None:
    parser = argparse.ArgumentParser("RDDM GT-RAIN training (ablation-ready, Windows-safe)")

    # Core
    parser.add_argument("--sampling_timesteps", type=int, default=10)
    parser.add_argument("--train_num_steps", type=int, default=120000)
    parser.add_argument("--save_and_sample_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=10)

    # GPU / Perf
    parser.add_argument("--device", type=str, default="0", help="CUDA_VISIBLE_DEVICES, e.g. '0'")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (推薦，省 VRAM)")
    parser.add_argument("--num_workers", type=int, default=0, help="Windows 建議 0 避免 multiprocessing 問題")

    # Data / model
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--train_lr", type=float, default=8e-5)

    # Ablation knobs (要接到 src 的 schedule 才會真的生效)
    parser.add_argument("--beta_schedule", type=str, default="linear",
                        choices=["linear", "cosine", "sigmoid", "quadratic"])
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--beta_scale", type=float, default=1.0)

    # Run management
    parser.add_argument("--run_name", type=str, default="", help="空白 => 自動命名")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--resume", type=str, default="none", choices=["none", "latest"])
    parser.add_argument("--resume_milestone", type=int, default=None, help="指定 milestone，例如 80")
    parser.add_argument("--skip_test", action="store_true", help="只 train 不 test")

    args = parser.parse_args()

    # Windows: 避免 dataloader 多工炸掉
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.stdout.flush()
    set_seed(args.seed)

    # Data flist（你已經換成 data_gtrain）
    HERE = Path(__file__).resolve().parent
    folder = [
        str(HERE / "data_gtrain" / "train_gt.flist"),
        str(HERE / "data_gtrain" / "train_input.flist"),
        str(HERE / "data_gtrain" / "test_gt.flist"),
        str(HERE / "data_gtrain" / "test_input.flist"),
    ]

    # Model setting（保持你目前 restoration 模式）
    original_ddim_ddpm = False
    condition = True
    input_condition = False
    input_condition_mask = False

    num_samples = 1
    sum_scale = 1

    if original_ddim_ddpm:
        model = Unet(dim=64, dim_mults=(1, 2, 4, 8))
        diffusion = GaussianDiffusion(
            model,
            image_size=args.image_size,
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
            input_condition=input_condition
        )

        # 嘗試把 beta_* 參數傳進 ResidualDiffusion（若 src 還沒支援，會直接報錯提醒你去 patch）
        rd_kwargs = dict(beta_schedule=args.beta_schedule, beta_end=args.beta_end, beta_scale=args.beta_scale)
        sig = inspect.signature(ResidualDiffusion.__init__)
        supported = set(sig.parameters.keys())
        filtered = {k: v for k, v in rd_kwargs.items() if k in supported}

        if (args.beta_end != 0.02 or args.beta_scale != 1.0 or args.beta_schedule != "linear") and len(filtered) < len(rd_kwargs):
            missing = sorted(set(rd_kwargs.keys()) - set(filtered.keys()))
            raise RuntimeError(
                "ResidualDiffusion 目前不支援你指定的 beta schedule 參數（缺少："
                + ", ".join(missing)
                + "）。\n"
                "請先在 src/residual_denoising_diffusion_pytorch.py 把 schedule 參數接進去。"
            )

        diffusion = ResidualDiffusion(
            model,
            image_size=args.image_size,
            timesteps=1000,
            sampling_timesteps=args.sampling_timesteps,
            objective="pred_res_noise",
            loss_type="l1",
            condition=condition,
            sum_scale=sum_scale,
            input_condition=input_condition,
            input_condition_mask=input_condition_mask,
            **filtered,
        )

    # run_name / results
    if not args.run_name:
        args.run_name = _build_run_name(args)

    results_root = HERE / args.results_root
    run_dir = results_root / args.run_name
    _safe_mkdir(run_dir)

    # metadata（方便最後彙整）
    meta = {
        "run_name": args.run_name,
        "seed": args.seed,
        "train_num_steps": args.train_num_steps,
        "sampling_timesteps": args.sampling_timesteps,
        "beta_schedule": args.beta_schedule,
        "beta_end": args.beta_end,
        "beta_scale": args.beta_scale,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "train_lr": args.train_lr,
        "amp": args.amp,
        "device": args.device,
        "num_workers": args.num_workers,
        "git_commit": _git_commit_short(),
        "gpu": _gpu_name(),
        "torch_version": _torch_version(),
        "platform": platform.platform(),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # Trainer
    trainer_kwargs = dict(
        train_batch_size=args.batch_size,
        num_samples=num_samples,
        train_lr=args.train_lr,
        train_num_steps=args.train_num_steps,
        gradient_accumulate_every=args.grad_accum,
        ema_decay=0.995,
        amp=args.amp,
        convert_image_to="RGB",
        condition=condition,
        save_and_sample_every=args.save_and_sample_every,
        equalizeHist=False,
        crop_patch=False,
        generation=False,
    )

    # 如果你們 Trainer 支援 num_workers，就傳；不支援就略過（避免你改壞 src）
    t_sig = inspect.signature(Trainer.__init__)
    if "num_workers" in t_sig.parameters:
        trainer_kwargs["num_workers"] = args.num_workers

    trainer = Trainer(diffusion, folder, **trainer_kwargs)

    # 把輸出 folder 綁在 run_dir（避免不同人互相覆蓋）
    if hasattr(trainer, "set_results_folder"):
        trainer.set_results_folder(str(run_dir))

    # ✅ 安全 resume：檔案存在才 load
    if trainer.accelerator.is_local_main_process:
        milestone = None
        if args.resume_milestone is not None:
            milestone = args.resume_milestone
        elif args.resume == "latest":
            milestone = _find_latest_milestone(run_dir)

        if milestone is not None:
            try:
                trainer.load(milestone)
                print(f"[INFO] Resumed from milestone={milestone}")
            except FileNotFoundError:
                print(f"[WARN] Requested resume milestone={milestone} but checkpoint not found; training from scratch.")
        else:
            print("[INFO] No resume; training from scratch.")

    # Train
    trainer.train()

    if args.skip_test:
        return

    # Test（只在 main process 跑）
    if trainer.accelerator.is_local_main_process:
        last_ms = args.train_num_steps // args.save_and_sample_every
        try:
            trainer.load(last_ms)
        except FileNotFoundError:
            latest = _find_latest_milestone(run_dir)
            if latest is None:
                raise RuntimeError("No checkpoint found for testing.")
            trainer.load(latest)
            last_ms = latest

        test_dir = run_dir / f"test_timestep_{args.sampling_timesteps}" / f"ckpt_{last_ms}"
        _safe_mkdir(test_dir)

        if hasattr(trainer, "set_results_folder"):
            trainer.set_results_folder(str(test_dir))

        trainer.test(last=True)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
