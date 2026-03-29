import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from modified_UNet.dataset import Data, InverseDataset
from modified_UNet.UNet import UNetModified


digit = 1       # 第一版数据
model_save = 1      # 保存 best_model_metrics 第几版的 的历史数据

# epoch 为 203 时的参数设置
batch_size = 8
epochs = 203
lr = 3e-4

num_workers = min(8, os.cpu_count() or 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = os.path.join(os.path.dirname(__file__), "save_metrics", f"history_{model_save}")

def save_history_npz(history, save_path):
    os.makedirs(save_dir, exist_ok=True)  # 自动创建目录
    file_path = os.path.join(save_dir, save_path + ".npz")
    np.savez(
        file_path,
        epoch=np.array(history["epoch"], dtype=np.int32),
        train_loss=np.array(history["train_loss"], dtype=np.float32)
    )


def save_history_csv(history, save_path):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, save_path + ".csv")

    with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss"])
        for row in zip(history["epoch"], history["train_loss"]):
            writer.writerow(row)


if __name__ == '__main__':
    print(f"Using device: {device}")

    # Step1 加载数据
    input_file = f"chi_all_{digit:02d}.npz"
    output_file = f"sample_all_{digit:02d}.npz"

    train_input, train_output, test_input, test_output = Data.get_data(input_file, output_file, 0.1)

    # Step2 创建Dataset
    train_dataset = InverseDataset(train_input, train_output)

    # Step3 创建DataLoader
    # 接收 dataset，而不是在里面重新创建（只负责包装 DataLoader）
    # 将数据按照batch的大小进行分组：batch1  batch2  batch3  ...
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    print(f"Train samples: {len(train_dataset)}")

    model = UNetModified().to(device)       # Step4 初始化网络
    criterion = nn.MSELoss()                # Step5 loss 定义损失函数
    optimizer = optim.Adam(model.parameters(), lr=3e-4)   # Step6 Optimizer,使用 Adam
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)     # Scheduler

    history = {
        "epoch": [],
        "train_loss": []
    }

    best_train_loss = float("inf")

    for epoch in range(epochs):         # Step7 训练
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            prediction = model(x)["out"]
            loss = criterion(prediction, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()    # 新增：每个epoch结束后更新学习率（必须放在这里）

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)

        save_history_npz(history, f"history_metrics_{model_save}")
        save_history_csv(history, f"history_metrics_{model_save}")        # 本质是 表格数据，可以使用Excel打开

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"Train Loss {train_loss:.6f} | "
            f"LR {optimizer.param_groups[0]['lr']:.2e}"
        )

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), f"best_model_{model_save}.pth")
            print("Model Saved")

    torch.save(model.state_dict(), f"last_model_{model_save}.pth")
    print("训练完成，历史指标已保存")
