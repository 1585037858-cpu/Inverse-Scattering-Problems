import os
import numpy as np
import torch
from torch.utils.data import Dataset


N_SAMPLES = 2000

class Data:     # 规格都是：(N, C, H ,W) 即：（2000，2，50，50） 跟（2000，1，50，50）

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    initial_guess_path = os.path.join(data_dir, f"guess_data_{N_SAMPLES}")
    scatterers_path = os.path.join(data_dir, f"scatterers_data_{N_SAMPLES}")

    @staticmethod
    def check_data_sanctity(input, output):
        assert not np.isnan(input).any()
        assert not np.isnan(output).any()

    @staticmethod
    def get_input_data_two(filename):       # filename: “chi_all_01.npz” 在 guess_path 路径下的所需 数据集 名称
        real_data = []
        imag_data = []
        filepath = os.path.join(Data.initial_guess_path, filename)
        initial_data = np.load(filepath, allow_pickle=True)
        initial_guesses = initial_data["chi"]       # chi 格式 ： （N_SAMPLES, 2, 50, 50） 内部存储的都为实数，即：将 虚部 在 实部 之后
        for guess in initial_guesses:
            real_data.append(guess[0])
            imag_data.append(guess[1])
        return real_data, imag_data     # 转换为 chi 的 实部 跟 虚部 返回 （N_SAMPLES, 50, 50）

    @staticmethod
    def get_input_data_one(filename):
        filepath = os.path.join(Data.initial_guess_path, filename)
        initial_data = np.load(filepath, allow_pickle=True)
        initial_guesses = initial_data["chi"]  # chi 格式 ： （N_SAMPLES, 2, 50, 50） 内部存储的都为实数，即：将 虚部 在 实部 之后

        return initial_guesses

    @staticmethod
    def get_output_data(filename):       # filename: “sample_all_01.npz” 在 scatterers_path 路径下的所需 数据集 名称
        filepath = os.path.join(Data.scatterers_path, filename)
        scatterer_data = np.load(filepath, allow_pickle=True)
        scatterers_original = scatterer_data["scatterers_inverse"]
        scatterers_original = np.real(scatterers_original)
        scatterers_original = np.expand_dims(scatterers_original, axis=1)

        return scatterers_original      # 返回 散射体 的 介电常数值（N_SAMPLES, 1, 50, 50）的实部

    @staticmethod
    def split_data(input, output, test_size=0.1):
        test_data_len = int(input.shape[0] * test_size)
        train_data_len = input.shape[0] - test_data_len

        input = np.asarray(input)       # 将 input跟output 的 数据格式 转换为 NumPy数组
        output = np.asarray(output)

        train_input, train_output = input[:train_data_len, :, :, :], output[:train_data_len, :, :, :]
        test_input, test_output = input[train_data_len:, :, :, :], output[train_data_len:, :, :, :]
        return train_input, train_output, test_input, test_output

    @staticmethod     # 只有 训练集 跟 测试集 时
    def get_data(input_path, output_path, test_size=0.1):
        x = Data.get_input_data_one(input_path)
        x = np.asarray(x)

        y = Data.get_output_data(output_path)
        y = np.asarray(y)

        Data.check_data_sanctity(x, y)

        train_input, train_output, test_input, test_output = Data.split_data(x, y, test_size)
        return train_input, train_output, test_input, test_output




class InverseDataset(Dataset):
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        x = self.input[idx]
        y = self.output[idx]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y


if __name__ == '__main__':

    digit = 1

    input_file = f"chi_all_{digit:02d}.npz"
    output_file = f"sample_all_{digit:02d}.npz"

    train_input, train_output, test_input, test_output = Data.get_data(input_file, output_file, 0.1)

    print("Training data input shape: ", train_input.shape)
    print("Test data input shape: ", test_input.shape)

    print("Training data output shape: ", train_output.shape)
    print("Test data output shape: ", test_output.shape)


    train_dataset = InverseDataset(train_input, train_output)

    print("Training dataset length: ", len(train_dataset))
