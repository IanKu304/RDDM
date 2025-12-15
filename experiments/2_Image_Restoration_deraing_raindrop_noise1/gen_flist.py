import os
import glob

# 1. 設定你的資料集根目錄 (Docker 內的絕對路徑)
dataset_root = "/app/experiments/2_Image_Restoration_deraing_raindrop_noise1/datasets/GT-RAIN"
train_path = os.path.join(dataset_root, "GT-RAIN_train")
# [修改] 改用 val 資料夾
val_path = os.path.join(dataset_root, "GT-RAIN_val") 

def generate_flist(source_path, prefix):
    gt_list = []
    input_list = []
    
    # 確保資料夾存在
    if not os.path.exists(source_path):
        print(f"Error: Path not found {source_path}")
        return

    # 遍歷所有場景資料夾
    for scene in sorted(os.listdir(source_path)):
        scene_dir = os.path.join(source_path, scene)
        if not os.path.isdir(scene_dir):
            continue
            
        # 找出該場景下所有的圖片
        images = sorted(glob.glob(os.path.join(scene_dir, "*.png")) + glob.glob(os.path.join(scene_dir, "*.jpg")))
        
        # 分類 GT (C-開頭) 和 Input (R-開頭)
        clears = [img for img in images if "C-" in os.path.basename(img)]
        rainys = [img for img in images if "R-" in os.path.basename(img)]
        
        if not clears:
            print(f"Warning: No Clear image found in {scene}")
            continue
            
        # 通常一個場景只有一張 Clear (GT)，但有多張 Rainy
        gt_img = clears[0] 
        
        for rainy_img in rainys:
            gt_list.append(gt_img)
            input_list.append(rainy_img)
            
    # 寫入檔案
    gt_flist_path = os.path.join(dataset_root, f"{prefix}_gt.flist")
    input_flist_path = os.path.join(dataset_root, f"{prefix}_input.flist")
    
    with open(gt_flist_path, 'w') as f:
        f.write('\n'.join(gt_list))
        
    with open(input_flist_path, 'w') as f:
        f.write('\n'.join(input_list))
        
    print(f"Generated {prefix} lists:")
    print(f"  GT: {len(gt_list)} images -> {gt_flist_path}")
    print(f"  Input: {len(input_list)} images -> {input_flist_path}")

# 執行生成
if __name__ == "__main__":
    print("Generating Training Lists...")
    generate_flist(train_path, "train")
    
    print("Generating Validation Lists...")
    # [修改] 生成 val 的清單
    generate_flist(val_path, "val")