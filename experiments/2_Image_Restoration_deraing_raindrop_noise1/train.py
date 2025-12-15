import os
import sys

from src.denoising_diffusion_pytorch import GaussianDiffusion
from src.residual_denoising_diffusion_pytorch import (ResidualDiffusion,
                                                      Trainer, Unet, UnetRes,
                                                      set_seed)

# ==========================================
#  新增一個小工具來讀取 .flist 檔案
# ==========================================
def read_flist(flist_path):
    """讀取 flist 檔案並回傳圖片路徑列表"""
    if not os.path.exists(flist_path):
        raise FileNotFoundError(f"找不到檔案: {flist_path}")
    
    with open(flist_path, 'r') as f:
        # 讀取每一行並去除前後空白
        lines = [line.strip() for line in f.readlines()]
    
    print(f"✅ 已讀取 {len(lines)} 張圖片路徑自: {os.path.basename(flist_path)}")
    return lines

# init 
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in [0])
sys.stdout.flush()
set_seed(10)
debug = False
if debug:
    save_and_sample_every = 2
    sampling_timesteps = 10
    sampling_timesteps_original_ddim_ddpm = 10
    train_num_steps = 200
else:
    save_and_sample_every = 1000
    if len(sys.argv)>1:
        sampling_timesteps = int(sys.argv[1])
    else:
        sampling_timesteps = 10
    sampling_timesteps_original_ddim_ddpm = 250
    train_num_steps = 120000

original_ddim_ddpm = False
if original_ddim_ddpm:
    condition = False
    input_condition = False
    input_condition_mask = False
else:
    condition = True
    input_condition = False
    input_condition_mask = False

if condition:
    if input_condition:
        folder = ["/home/liu/disk12t/liu_data/shadow_removal_with_val_dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_train.flist",
                "/home/liu/disk12t/liu_data/shadow_removal_with_val_dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_train.flist",
                "/home/liu/disk12t/liu_data/shadow_removal_with_val_dataset/ISTD_Dataset_arg/data_val/ISTD_mask_train.flist",
                "/home/liu/disk12t/liu_data/shadow_removal_with_val_dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_test.flist",
                "/home/liu/disk12t/liu_data/shadow_removal_with_val_dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_test.flist",
                "/home/liu/disk12t/liu_data/shadow_removal_with_val_dataset/ISTD_Dataset_arg/data_val/ISTD_mask_test.flist"]
    else:
        # folder = ["/home/ljw/data-set/program/diffusion/dataset/deraing_raindrop/raindrop_data/train_gt.flist",
        #         "/home/ljw/data-set/program/diffusion/dataset/deraing_raindrop/raindrop_data/train_input.flist",
        #         "/home/ljw/data-set/program/diffusion/dataset/deraing_raindrop/raindrop_data/test_gt.flist",
        #         "/home/ljw/data-set/program/diffusion/dataset/deraing_raindrop/raindrop_data/test_input.flist"]# [修改] 這裡改用剛剛生成的 val_gt.flist 和 val_input.flist
        base_path = "/app/experiments/2_Image_Restoration_deraing_raindrop_noise1/datasets/GT-RAIN"
        
        # 【關鍵修改】: 先用 read_flist 把路徑讀出來變成 list
        print("正在讀取資料集清單...")
        train_gt_list = read_flist(os.path.join(base_path, "train_gt.flist"))
        train_input_list = read_flist(os.path.join(base_path, "train_input.flist"))
        val_gt_list = read_flist(os.path.join(base_path, "val_gt.flist"))
        val_input_list = read_flist(os.path.join(base_path, "val_input.flist"))

        # 把讀好的圖片列表(list) 傳給 folder，而不是傳檔案路徑(string)
        folder = [
            train_gt_list,     # 訓練集 GT (圖片列表)
            train_input_list,  # 訓練集 Input (圖片列表)
            val_gt_list,       # 驗證集 GT
            val_input_list     # 驗證集 Input
        ]

    train_batch_size = 2
    num_samples = 1
    sum_scale = 1
    image_size = 128
else:
    folder = '/home/liu/disk12t/liu_data/dataset/CelebA/img_align_celeba'
    train_batch_size = 32
    num_samples = 25
    sum_scale = 1
    image_size = 32

if original_ddim_ddpm:
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    )
    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,           # number of steps
        sampling_timesteps=sampling_timesteps_original_ddim_ddpm,
        loss_type='l1',            # L1 or L2
    )
else:
    model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        share_encoder=0,
        condition=condition,
        input_condition=input_condition
    )
    diffusion = ResidualDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,           # number of steps
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        sampling_timesteps=sampling_timesteps,
        objective='pred_res_noise',
        loss_type='l1',            # L1 or L2
        condition=condition,
        sum_scale = sum_scale,
        input_condition=input_condition,
        input_condition_mask=input_condition_mask
    )

trainer = Trainer(
    diffusion,
    folder,
    train_batch_size=train_batch_size,
    num_samples=num_samples,
    train_lr=8e-5,
    train_num_steps=train_num_steps,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=False,                        # turn on mixed precision
    convert_image_to="RGB",
    condition=condition,
    save_and_sample_every=save_and_sample_every,
    equalizeHist=False,
    crop_patch=False,
    generation = False
)

if not trainer.accelerator.is_local_main_process:
    pass
else:
    # trainer.load(80)  <-- 註解掉這一行，因為我們是從頭訓練
    print("Starting new training...")

# train
trainer.train()

# test
if not trainer.accelerator.is_local_main_process:
    pass
else:
    trainer.load(trainer.train_num_steps//save_and_sample_every)
    trainer.set_results_folder('./results/test_timestep_'+str(sampling_timesteps))
    trainer.test(last=True)

# trainer.set_results_folder('./results/test_sample')
# trainer.test(sample=True)
