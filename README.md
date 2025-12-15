# Residual Diffusion Denoising Model (RDDM)

This repository contains official code for the Residual Diffusion Denoising Model (RDDM) experiments, and **our group’s reproduction pipeline** for **GT-RAIN** (deraining) with **alpha/beta schedule ablation** and **FID comparison**.

> Notes for our course / group report:
> - We **do not commit** datasets, checkpoints, or experiment results to GitHub.
> - Everyone downloads the shared `data.zip` from Drive and **unzips to `RDDM/data/`** locally.
> - We recommend **Windows + Anaconda** setup below (works with CUDA 11.8 PyTorch wheels).

---

## 0) Git & collaboration (CLI + GitHub Desktop)

### 0.1 Clone (CLI)
```bash
git clone [<YOUR_FORK_URL>](https://github.com/IanKu304/RDDM)
cd RDDM
git remote add upstream <ORIGINAL_RDDM_URL>
git fetch upstream
```

### 0.2 Clone (GitHub Desktop)
1. **File → Clone repository…**
2. Select your fork, choose local path, click **Clone**.
3. If you need to add `upstream`: **Repository → Repository settings → Remote**.

### 0.3 Ignore datasets / checkpoints (required)
We keep datasets under `RDDM/data/` and ignore them via `.gitignore`.

Recommended `.gitignore` (already validated: `git status --ignored` shows `data/` as ignored):
```gitignore
# Ignore Datasets
*.zip
*.tar.gz
data/
datasets/GT-RAIN/
experiments/2_Image_Restoration_deraing_raindrop_noise1/data/

# Ignore Checkpoints and Logs
*.pth
*.pt
*.log
experiments/**/results/
experiments/**/training.log
```

Commit `.gitignore` changes:

**CLI**
```bash
git add .gitignore
git commit -m "Update .gitignore: ignore datasets, checkpoints, logs"
git push
```

**GitHub Desktop**
1. Open repo → **Changes**
2. Stage `.gitignore`
3. Commit message → **Commit to main**
4. **Push origin**

---

## 1) Environment setup (Windows + Anaconda, recommended)

### 1.1 Create environment
From repo root:
```bash
conda env create -f install_win.yaml -n rddm
conda activate rddm
python -m pip install -U pip setuptools wheel
```

### 1.2 Install PyTorch (CUDA 11.8)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify:
```bash
python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available())"
```

### 1.3 Known missing packages (if you see ModuleNotFoundError)
```bash
pip install ema-pytorch Augmentor lmdb
```

Optional (IPython warnings on Windows):
```bash
pip install colorama exceptiongroup
```

---

## 2) Dataset (GT-RAIN)

### 2.1 Download
- Official page: https://visual.ee.ucla.edu/gt_rain.htm/
- We used HF mirror:
```bash
git clone https://huggingface.co/datasets/hwdz15508/GT-RAIN
```

### 2.2 Place dataset under `RDDM/data/`
Your local structure should look like:
```
RDDM/
  data/
    GT-RAIN_train/   (unzipped)
    GT-RAIN_val/     (unzipped)
    GT-RAIN_test/    (unzipped)
```

Group workflow:
1. One person unzips everything into `RDDM/data/`
2. Zip the `data/` folder → upload to Drive
3. Everyone else downloads and unzips into their own `RDDM/data/`

### 2.3 Create paired filelists (.flist)
The deraining experiment reads paired lists:
- `train_input.flist`: rainy image paths
- `train_gt.flist`: clean image paths
- `test_input.flist`, `test_gt.flist`

Create:
```
RDDM/data/gt_rain_flists/
  train_input.flist
  train_gt.flist
  test_input.flist
  test_gt.flist
```

**Suggested script (save as `tools/make_gtrain_flists.py`)**:
```python
import argparse, os
from pathlib import Path

def list_pngs(p): return sorted([x for x in Path(p).rglob("*.png")])

def main(data_root, out_dir, train_dir="GT-RAIN_train", test_dir="GT-RAIN_test"):
    data_root = Path(data_root)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    def build_split(split_dir):
        rainy = []
        clean = []

        for img in list_pngs(data_root / split_dir):
            name = img.name
            # rainy frames contain "-R-"
            if "-R-" in name:
                rainy.append(str(img))
                # training/test: assume clean is "*-C-000.png" in the same folder
                clean_name = name.replace("-R-", "-C-").split("-C-")[0] + "-C-000.png"
                clean_path = img.with_name(clean_name)
                clean.append(str(clean_path))
        return rainy, clean

    tr_in, tr_gt = build_split(train_dir)
    te_in, te_gt = build_split(test_dir)

    (out_dir / "train_input.flist").write_text("\n".join(tr_in), encoding="utf-8")
    (out_dir / "train_gt.flist").write_text("\n".join(tr_gt), encoding="utf-8")
    (out_dir / "test_input.flist").write_text("\n".join(te_in), encoding="utf-8")
    (out_dir / "test_gt.flist").write_text("\n".join(te_gt), encoding="utf-8")

    print("Saved:", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out_dir", default="data/gt_rain_flists")
    args = ap.parse_args()
    main(args.data_root, args.out_dir)
```

Run from repo root:
```bash
python tools/make_gtrain_flists.py --data_root data --out_dir data/gt_rain_flists
```

---

## 3) Configure experiment path (IMPORTANT)

Go to:
```
experiments/2_Image_Restoration_deraing_raindrop_noise1/train.py
```

Find the `folder = [...]` list and replace it with your local `.flist` paths.

Recommended Windows-safe version:
```python
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FLIST_DIR = os.path.join(ROOT, "data", "gt_rain_flists")

folder = [
    os.path.join(FLIST_DIR, "train_gt.flist"),
    os.path.join(FLIST_DIR, "train_input.flist"),
    os.path.join(FLIST_DIR, "test_gt.flist"),
    os.path.join(FLIST_DIR, "test_input.flist"),
]
```

---

## 4) Windows multiprocessing fix (MUST do before training)

On Windows, PyTorch DataLoader uses `spawn`. If `train.py` is not guarded, you may see:

> RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase…

Fix: wrap the training entrypoint with `if __name__ == "__main__":`.

At the bottom of `train.py`, change to:

```python
def main():
    # ---- keep the original code that builds model/trainer here ----
    trainer.train()

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
```

If you still get worker crashes, set DataLoader workers to 0 (search in `src/residual_denoising_diffusion_pytorch.py` for `DataLoader(` and set `num_workers=0` for Windows).

---

## 5) Train (Deraining, GT-RAIN)

From the experiment folder:
```bash
cd experiments/2_Image_Restoration_deraing_raindrop_noise1
python train.py 10
```

- The `10` is typically used as **sampling timesteps** in the provided script.
- Batch size is defined in `train.py` via `train_batch_size = ...`.

Outputs are saved under the experiment’s `results/` folder (ignored by git).

---

## 6) Evaluate / Test (optional)
If the experiment includes `test.py`:
```bash
python test.py
```

For metrics:
- PSNR / SSIM scripts are under `eval/deraining_eval/`
- FID script is under `eval/image_generation_eval/fid_and_inception_score.py`

---

## 7) Common troubleshooting

### 7.1 `PackagesNotFoundError` when `conda env create -f install.yaml`
Cause: original `install.yaml` is a **Linux-pinned** environment file; use `install_win.yaml` on Windows.

### 7.2 `No module named 'ema_pytorch' / 'Augmentor' / 'lmdb'`
```bash
pip install ema-pytorch Augmentor lmdb
```

### 7.3 NumPy `np.str` warning / incompatibility
If you see warnings (or errors on NumPy 2.x), prefer:
- Pin NumPy `< 2` (already in `install_win.yaml`), or
- Patch `datasets/base.py`: replace `dtype=np.str` with `dtype=str` or `dtype=np.str_`.

### 7.4 DataLoader worker exited unexpectedly (Windows)
Apply Section 4 fix, then retry. If needed:
- set `num_workers=0`.

---

## 8) Reproduction: alpha/beta schedule ablation + FID
For our report, we vary:
- **alpha schedule** and **beta schedule** (see the scheduler / diffusion config in `src/*diffusion*.py`)
- For each setting: train → sample → compute FID with `eval/image_generation_eval/fid_and_inception_score.py`

We log:
- config (alpha/beta type + key hyperparameters)
- checkpoint id / step
- FID score (mean over runs if repeated)
