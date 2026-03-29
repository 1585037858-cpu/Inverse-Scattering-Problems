import os
import numpy as np
from scatterer import Scatterer

from utils.plot_utils import PlotUtils

# === 参数设置 ===
WAVE_LENGTH = 0.125
N_SAMPLES = 2000          # 样本数量（论文使用 5000）
PROBLEM_TYPE = "forward"  # 使用前向问题生成数据
SHAPES = ["circle", "square"]
SIZES = [0.5*WAVE_LENGTH, WAVE_LENGTH, 1.5*WAVE_LENGTH, 2*WAVE_LENGTH, 2.5*WAVE_LENGTH]

# 介电常数区间
# EPSILON_LOW = [1, 5]
# EPSILON_HIGH = [50, 77]
EPSILON_LOW = np.arange(1, 5.1, 0.2)
EPSILON_HIGH = np.arange(50, 77.1, 0.2)

ROOT_DIR = "..\data"
OUTPUT_DIR = os.path.join(ROOT_DIR, f"scatterers_data_{N_SAMPLES}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def random_scatterer_params():
    """生成两个随机散射体的参数"""
    scatterer_params = []
    for i in range(2):
        shape = np.random.choice(SHAPES)
        center_x = np.round(np.random.uniform(-0.6, 0.6), 2)     # 在该区间均匀采样，结果为浮点数
        center_y = np.round(np.random.uniform(0.15, 0.6), 2)
        size = np.random.choice(SIZES) / 2      # 求其半径
        if i == 0:
            eps_r = np.round(np.random.choice(EPSILON_LOW), 1)  # 保留1位小数
            # eps_r = np.random.uniform(*EPSILON_LOW)     # 第一个散射体是：低介电常数
        else:
            eps_r = np.round(np.random.choice(EPSILON_HIGH), 1)  # 保留1位小数
            # eps_r = np.random.uniform(*EPSILON_HIGH)
        eps_i = eps_r * 0.1       # 无损耗模式
        # eps_i = 0       # 无损耗模式
        # eps_i = np.random.uniform(0.1, 0.3) * eps_r * 0.1  # 模拟损耗
        scatterer_params.append({
            "shape": shape,
            "center_x": center_x,
            "center_y": center_y,
            "size": size,
            "permittivity": eps_r + 1j * eps_i
        })
    return scatterer_params


def generate_dataset(n_samples=N_SAMPLES):

    permittivity_forward = []
    permittivity_inverse = []

    """批量生成训练样本并保存"""
    for i in range(n_samples):
        params = random_scatterer_params()      # 生成散射体相关参数

        # 前向问题：生成介电常数系数 （400×400）
        scatterer_forward = Scatterer(problem=PROBLEM_TYPE, inverse_type=None, scatterer_params=params).generate()
        permittivity_forward.append(scatterer_forward)

        # 反向问题：生成介电常数系数 (50×50)
        scatterer_inverse = Scatterer(problem="inverse", inverse_type=None, scatterer_params=params).generate()
        permittivity_inverse.append(scatterer_inverse)


    # 保存为 npz 文件 复数（实部 + 虚部）
    filename = os.path.join(OUTPUT_DIR, f"sample_all_01.npz")
    np.savez(
        filename,
        scatterers_forward = permittivity_forward,
        scatterers_inverse = permittivity_inverse,
        scatterer_params = params       # 参数格式是什么？两个二维数组，
        # {'center_x': 0.4807545296848422, 'center_y': 0.4059793459637403, 'permittivity': (2.8+0j), 'shape': square, 'size': 0.125}
        # {'center_x': 0.2904241923732649, 'center_y': 0.5169989638463486, 'permittivity': (56.20000000000009+0j), 'shape': square, 'size': 0.15625}
    )
    if i % 100 == 0:
            print(f"Generated {i}/{n_samples} samples")

    print(f"\n✅ 数据生成完成，共保存 {n_samples} 个样本到 {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_dataset(N_SAMPLES)

    # 加载.npz文件
    filename = os.path.join(OUTPUT_DIR, f"sample_all_01.npz")
    sample = np.load(filename, allow_pickle=True)
    scatterer_forward = sample["scatterers_forward"][1]       # 格式为：（400，400）
    scatterer_inverse = sample["scatterers_inverse"][1]       # 格式为：（50，50）

    PlotUtils.view_scatterer(scatterer_forward, scatterer_inverse)

    sample.close()
