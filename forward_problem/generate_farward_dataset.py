import os
import numpy as np
from tqdm import tqdm
from forward_problem.solve import ForwardProblemSolver

# =========================
# 参数设置
# =========================
N_SAMPLES = 2000          # 样本数量（论文使用 5000）

INPUT_DIR = F"../data/scatterers_data_{N_SAMPLES}"     # 散射体相关参数(形状、大小、介电常数) 数据
OUTPUT_DIR = f"../data/field_data_{N_SAMPLES}"     # 散射体产生的各种场、功率 数据
os.makedirs(OUTPUT_DIR, exist_ok=True)



if __name__ == "__main__":

    # 加载.npz文件
    filename = os.path.join(INPUT_DIR, "sample_all_01.npz")
    data = np.load(filename, allow_pickle=True)
    scatterers = data["scatterers_forward"]       # 格式为：（2000,400，400）

    with tqdm(total=len(scatterers), desc="Running forward simulations") as pbar:
        for i, scatterer in enumerate(scatterers):  # 循环处理每个散射体，打印 i（0~9）

            # 对每个单独的介电常数数据 进行 实例化前向仿真求解器
            solver = ForwardProblemSolver(scatterer)
            direct_field, direct_power, scattered_field, total_field, total_power = solver.generate_forward_data()

            # 保存为 npz 文件
            filename = os.path.join(OUTPUT_DIR, f"sample_all_{i:04d}.npz")

            np.savez(
                filename,
                scatterer = scatterer,
                direct_field = direct_field,
                direct_power = direct_power,
                scattered_field = scattered_field,
                total_field = total_field,
                total_power = total_power,
            )

            pbar.update(1)

    # 随机展示一个样本
    ForwardProblemSolver.get_field_plots(total_field, direct_field, scattered_field, 39)

