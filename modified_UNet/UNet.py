# unet_pytorch.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """ (Conv2d → BatchNorm → ReLU) """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1): # 默认 kernel_size=3，padding=1 SAME (即：尺寸不变)
        super().__init__()

        self.one_cnn_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.one_cnn_layer(x)


# 完整的U-Net网络结构
class UNetModified(nn.Module):
    def __init__(self):       # base_c: 基础滤波器的数量： 第一个编码层的通道数
        # 调用父类的构造函数
        super().__init__()

        # -------- Encoder --------

        self.conv1_1 = ConvBlock(2, 64, 3, 0)     # VALID   48×48
        self.conv1_2 = ConvBlock(64, 64)    # SAME   48×48
        self.conv1_3 = ConvBlock(64, 64)    # SAME   48×48


        self.pool1 = nn.MaxPool2d(2, 2)    # 24×24

        self.conv2_1 = ConvBlock(64, 128)    # 24×24
        self.conv2_2 = ConvBlock(128, 128)   # 24×24
        self.conv2_3 = ConvBlock(128, 128)   # 24×24

        self.pool2 = nn.MaxPool2d(2, 2)    # 12×12

        self.conv3_1 = ConvBlock(128, 256)  # 12×12
        self.conv3_2 = ConvBlock(256, 256)  # 12×12
        self.conv3_3 = ConvBlock(256, 256)  # 12×12

        self.pool3 = nn.MaxPool2d(2, 2)    # 6×6

        self.conv4_1 = ConvBlock(256, 512)  # 6×6
        self.conv4_2 = ConvBlock(512, 512)  # 6×6
        self.conv4_3 = ConvBlock(512, 512)  # 6×6

        # -------- Decoder --------

        # mode='nearest'：最近邻插值。 上采样本身不改变通道数 up5: (512, 12, 12)
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv5_1 = ConvBlock(512 + 256, 256, 2, 'same')
        self.conv5_2 = ConvBlock(256, 256)
        self.conv5_3 = ConvBlock(256, 256)

        self.up6 = nn.Upsample(scale_factor=2)

        self.conv6_1 = ConvBlock(256 + 128, 128, 2, 'same')
        self.conv6_2 = ConvBlock(128, 128)
        self.conv6_3 = ConvBlock(128, 128)

        self.up7 = nn.Upsample(scale_factor=2)

        self.conv7_1 = ConvBlock(128 + 64, 64, 2, 'same')
        self.conv7_2 = ConvBlock(64, 64)
        self.conv7_3 = ConvBlock(64, 64)

        # 恢复 50x50
        self.deconv = nn.ConvTranspose2d(64, 1, kernel_size=3)

        self.final = nn.Conv2d(1 + 2, 1, kernel_size=1)

    def forward(self, x):

        # Encoder
        c1 = self.conv1_1(x)
        c1 = self.conv1_2(c1)
        c1 = self.conv1_3(c1)
        p1 = self.pool1(c1)

        c2 = self.conv2_1(p1)
        c2 = self.conv2_2(c2)
        c2 = self.conv2_3(c2)
        p2 = self.pool2(c2)

        c3 = self.conv3_1(p2)
        c3 = self.conv3_2(c3)
        c3 = self.conv3_3(c3)
        p3 = self.pool3(c3)

        c4 = self.conv4_1(p3)
        c4 = self.conv4_2(c4)
        c4 = self.conv4_3(c4)

        # Decoder
        u5 = self.up5(c4)
        m5 = torch.cat([c3, u5], dim=1)

        c5 = self.conv5_1(m5)
        c5 = self.conv5_2(c5)
        c5 = self.conv5_3(c5)

        u6 = self.up6(c5)
        m6 = torch.cat([c2, u6], dim=1)

        c6 = self.conv6_1(m6)
        c6 = self.conv6_2(c6)
        c6 = self.conv6_3(c6)

        u7 = self.up7(c6)
        m7 = torch.cat([c1, u7], dim=1)

        c7 = self.conv7_1(m7)
        c7 = self.conv7_2(c7)
        c7 = self.conv7_3(c7)

        c8 = self.deconv(c7)

        m9 = torch.cat([x, c8], dim=1)

        out = self.final(m9)

        return {"out": F.relu(out)}


if __name__ == "__main__":
    model = UNetModified()
    input_tensor = torch.randn(1, 2, 50, 50)  # 输入大小
    output = model(input_tensor)
    print(output["out"].shape)
