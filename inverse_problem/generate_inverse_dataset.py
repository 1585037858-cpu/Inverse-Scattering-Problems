import os
import numpy as np
from tqdm import tqdm

from inverse_problem.solve import InverseProblemSolver

from utils.plot_utils import PlotUtils


# =============================
# 路径配置
# =============================
N_SAMPLES = 2000          # 样本数量（论文使用 5000）

ROOT_DIR = "../data"
TRAIN_DIR = os.path.join(ROOT_DIR, f"scatterers_data_{N_SAMPLES}")
FORWARD_DIR = os.path.join(ROOT_DIR, f"field_data_{N_SAMPLES}")
OUTPUT_DIR = os.path.join(ROOT_DIR, f"guess_data_{N_SAMPLES}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# xRA 数据生成流程
# =============================
def generate_xra_dataset(model_name, prior, solver_params = {"alpha": 0, "sparse": False}):
    """
    使用 InverseProblemSolver，通过扩展 Rytov 模型 (xRA)
    对前向仿真结果进行处理，生成网络训练输入 (对比度函数的实部与虚部：Re/Im χ_RI) 和输出 (介电常数：ε_R)
    """
    reconstruction = []

    filename = os.path.join(TRAIN_DIR, "sample_all_01.npz")     # 散射体 介电常数 数据
    data = np.load(filename, allow_pickle=True)
    scatterers = data["scatterers_forward"]       # 格式为：（2000,400，400） 为复数

    with tqdm(total=len(scatterers), desc="Running reconstruction simulations") as pbar:
        for i, scatterer in enumerate(scatterers):  # 循环处理每个散射体，打印 i（0~9）

            filepath = os.path.join(FORWARD_DIR, f"sample_all_{i:04d}.npz")     # 获取每个散射体对应的前向仿真 场、功率
            field_data = np.load(filepath, allow_pickle=True)

            # reconstruct profile from field data
            direct_power = field_data["direct_power"]
            total_power = field_data["total_power"]

            # 调用 InverseProblemSolver 求解 χ(对比度函数)
            model = InverseProblemSolver(direct_power, total_power, model_name, prior, solver_params)
            chi = model.solve()

            reconstruction.append(chi)

            pbar.update(1)

    # 保存为 npz 文件
    filename = os.path.join(OUTPUT_DIR, f"chi_all_01.npz")

    np.savez(filename, chi = reconstruction)


if __name__ == "__main__":

    # generate_xra_dataset(
    #     model_name="prytov_complex",  # 论文使用复对比度模型
    #     prior="qs2D",                # 对应 H1 或 Ridge 正则
    #     solver_params={"alpha": 8, "sparse": False}    # α：正则化参数，sparse：是否为稀疏性
    # )

    # 加载.npz文件
    number = 50

    # 加载 散射体 的介电常数值
    filename = f'../data/scatterers_data_{N_SAMPLES}/sample_all_01.npz'  # 请替换为实际文件名
    data = np.load(filename, allow_pickle=True)
    scatterers_inverse = data["scatterers_inverse"]
    data.close()

    # 加载 散射体 的对比度函数
    filename = f'../data/guess_data_{N_SAMPLES}/chi_all_01.npz'  # 请替换为实际文件名
    data = np.load(filename, allow_pickle=True)
    scatterers_guess = data["chi"]
    scatterers_guess = np.moveaxis(scatterers_guess, 1, -1)
    data.close()

    PlotUtils.check_data(scatterers_guess ,scatterers_inverse)
