import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modified_UNet.dataset import Data, InverseDataset
from modified_UNet.UNet import UNetModified
from modified_UNet.metrics import psnr

from utils.plot_utils import *

digit = 1
batch_size = 8
num_workers = 0
model_path = "best_model_1.pth"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================
    # Step1 读取数据
    # =========================
    input_file = f"chi_all_{digit:02d}.npz"
    output_file = f"sample_all_{digit:02d}.npz"

    train_input, train_output, test_input, test_output = Data.get_data(input_file, output_file, 0.1)

    print(f"Test input shape : {test_input.shape}")
    print(f"Test output shape: {test_output.shape}")

    # =========================
    # Step2 构建测试集
    # =========================
    test_dataset = InverseDataset(test_input, test_output)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    print(f"Test samples: {len(test_dataset)}")

    # =========================
    # Step3 加载模型
    # =========================
    model = UNetModified().to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model weights from: {model_path}")

    # =========================
    # Step4 定义损失函数
    # =========================
    criterion = nn.MSELoss()

    # =========================
    # Step5 测试集推理
    # =========================
    test_loss = 0.0
    test_psnr = 0.0

    all_inputs = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            prediction = model(x)["out"]
            loss = criterion(prediction, y)

            test_loss += loss.item()
            test_psnr += psnr(prediction, y)

            all_inputs.append(x.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_preds.append(prediction.cpu().numpy())

    test_loss /= len(test_loader)
    test_psnr /= len(test_loader)

    print(f"Test Loss : {test_loss:.6f}")
    print(f"Test PSNR : {test_psnr:.2f}")

    # 拼接全部结果，方便随机抽样
    all_inputs = np.concatenate(all_inputs, axis=0)    # [N, 2, H, W]
    all_targets = np.concatenate(all_targets, axis=0)  # [N, 1, H, W] or [N, H, W]
    all_preds = np.concatenate(all_preds, axis=0)      # [N, 1, H, W] or [N, H, W]

    # =========================
    # Step6 随机展示十个样本
    # =========================
    save_dir = "test_results"
    os.makedirs(save_dir, exist_ok=True)

    num_samples = 10
    for i in range(num_samples):
        idx = random.randint(0, len(all_inputs) - 1)
        print(f"Random sample index: {idx}")

        input_img = all_inputs[idx]
        target_img = all_targets[idx]
        pred_img = all_preds[idx]

        # 可选：输出该样本的单独指标
        target_tensor = torch.tensor(target_img).unsqueeze(0).float()
        pred_tensor = torch.tensor(pred_img).unsqueeze(0).float()
        sample_mse = nn.MSELoss()(pred_tensor, target_tensor).item()
        sample_psnr = psnr(pred_tensor, target_tensor)

        print(f"Sample MSE  : {sample_mse:.6f}")
        print(f"Sample PSNR : {sample_psnr:.2f}")


        save_path = os.path.join(save_dir, f"test_sample_{idx}.png")

        PlotUtils.plot_results(target_img, input_img, pred_img, save_path=save_path)


    # =========================
    # Step6 从100个样本中选出准确性最高的一个
    # =========================
    save_dir = "test_results"
    os.makedirs(save_dir, exist_ok=True)

    num_candidates = 200   # 从100个样本中挑选
    num_candidates = min(num_candidates, len(all_inputs))  # 防止测试集不足100

    # 随机抽取100个不同样本
    candidate_indices = random.sample(range(len(all_inputs)), num_candidates)

    best_idx = None
    best_mse = float("inf")
    best_psnr = -float("inf")

    for idx in candidate_indices:
        input_img = all_inputs[idx]
        target_img = all_targets[idx]
        pred_img = all_preds[idx]

        target_tensor = torch.tensor(target_img).unsqueeze(0).float()
        pred_tensor = torch.tensor(pred_img).unsqueeze(0).float()

        sample_mse = nn.MSELoss()(pred_tensor, target_tensor).item()
        sample_psnr = psnr(pred_tensor, target_tensor)

        print(f"Index {idx:4d} | MSE: {sample_mse:.6f} | PSNR: {sample_psnr:.2f}")

        # 以 MSE 最小 作为“准确性最高”
        if sample_mse < best_mse:
            best_mse = sample_mse
            best_psnr = sample_psnr
            best_idx = idx

    print("\n===== Best Sample Among 100 Candidates =====")
    print(f"Best index : {best_idx}")
    print(f"Best MSE   : {best_mse:.6f}")
    print(f"Best PSNR  : {best_psnr:.2f}")

    best_input = all_inputs[best_idx]
    best_target = all_targets[best_idx]
    best_pred = all_preds[best_idx]

    save_path = os.path.join(save_dir, f"best_sample_top1_from_{num_candidates}.png")
    PlotUtils.plot_results(best_target, best_input, best_pred, save_path=save_path)


if __name__ == "__main__":
    main()