import os
import shutil
from huggingface_hub import snapshot_download

# 目標路徑
local_dir = "./experiments/2_Image_Restoration_deraing_raindrop_noise1/datasets/GT-RAIN"

# 【關鍵修正】
# 在你的掛載目錄下建立一個暫存 cache 資料夾
# 這樣暫存檔和目標檔就在同一個硬碟上了
temp_cache_dir = "./experiments/temp_hf_cache"

print(f"Downloading to {local_dir} ...")

try:
    snapshot_download(
        repo_id="hwdz15508/GT-RAIN",
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        cache_dir=temp_cache_dir  # <--- 加了這一行，指定暫存位置
    )
    print("Download completed!")

finally:
    # 下載完後，把那個暫存資料夾刪掉，保持乾淨
    if os.path.exists(temp_cache_dir):
        print("Cleaning up temporary cache...")
        shutil.rmtree(temp_cache_dir)