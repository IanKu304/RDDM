import argparse
from pathlib import Path

def collect_pairs(split_dir: Path):
    """
    Return list of (clean_path, rainy_path) pairs for a GT-RAIN split directory.
    Strategy:
      - For each rainy: try matching clean with same index: *-R-XYZ.png -> *-C-XYZ.png
      - If not exist, fallback to *-C-000.png
    """
    pairs = []
    for scene_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        rainy_files = sorted(scene_dir.glob("*-R-*.png"))
        if not rainy_files:
            continue

        for r in rainy_files:
            # 1) try same index clean
            c_same = r.with_name(r.name.replace("-R-", "-C-"))
            if c_same.exists():
                c = c_same
            else:
                # 2) fallback to C-000
                prefix = r.name.split("-R-")[0]  # keep e.g. "...-Webcam"
                c_000 = r.with_name(prefix + "-C-000.png")
                c = c_000 if c_000.exists() else None

            if c is None or not c.exists():
                # Skip if cannot find clean
                continue

            pairs.append((c.resolve(), r.resolve()))
    return pairs

def write_flist(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in items:
            f.write(str(x) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True,
                    help="RDDM/data directory (contains GT-RAIN_train/val/test)")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output dir to write flists")
    ap.add_argument("--train_split", type=str, default="GT-RAIN_train",
                    choices=["GT-RAIN_train", "GT-RAIN_val", "GT-RAIN_test"])
    ap.add_argument("--test_split", type=str, default="GT-RAIN_val",
                    choices=["GT-RAIN_train", "GT-RAIN_val", "GT-RAIN_test"])
    ap.add_argument("--max_pairs", type=int, default=0,
                    help="Optional cap for quick debugging (0 = no cap)")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    train_pairs = collect_pairs(data_root / args.train_split)
    test_pairs  = collect_pairs(data_root / args.test_split)

    if args.max_pairs and args.max_pairs > 0:
        train_pairs = train_pairs[:args.max_pairs]
        test_pairs  = test_pairs[:args.max_pairs]

    # split into separate lists
    train_gt    = [c for (c, r) in train_pairs]
    train_input = [r for (c, r) in train_pairs]
    test_gt     = [c for (c, r) in test_pairs]
    test_input  = [r for (c, r) in test_pairs]

    write_flist(out_dir / "train_gt.flist", train_gt)
    write_flist(out_dir / "train_input.flist", train_input)
    write_flist(out_dir / "test_gt.flist", test_gt)
    write_flist(out_dir / "test_input.flist", test_input)

    print("[OK] train pairs:", len(train_pairs))
    print("[OK] test  pairs:", len(test_pairs))
    print("[OK] wrote flists to:", out_dir)

if __name__ == "__main__":
    main()
