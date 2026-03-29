import io
import random
import numpy as np
import matplotlib.pyplot as plt

from config import Config

class PlotUtils:

    @staticmethod
    def plot_setup():
        room_config = Config.room
        doi_config = Config.doi
        ratio = room_config["length"] / doi_config["length"]
        grids = int(doi_config["forward_grids"] * ratio)

        image = np.ones((grids, grids))
        if doi_config["origin"] == "center":
            diff = int((grids - doi_config["forward_grids"])/2)
            image[diff:grids-diff, diff:grids-diff] = 0
        else:
            diff = int(grids/2)
            image[diff:doi_config["forward_grids"], diff:doi_config["forward_grids"]] = 0
        plt.imshow(image, cmap=plt.cm.gray, extent=PlotUtils.get_room_extent())
        plt.show()

    @staticmethod
    def get_doi_extent():
        doi_length = Config.doi["length"]
        doi_width = Config.doi["width"]
        if Config.doi["origin"] == "center":
            extent = [-doi_length/2, doi_length/2, -doi_width/2, doi_width/2]
        else:
            extent = [0, doi_length, 0, doi_width]
        return extent

    @staticmethod
    def get_room_extent():
        room_length = Config.room["length"]
        room_width = Config.room["width"]
        if Config.room["origin"] == "center":
            extent = [-room_length / 2, room_length / 2, -room_width / 2, room_width / 2]
        else:
            extent = [0, room_length, 0, room_width]
        return extent

    @staticmethod
    def check_data(train_input, train_output):

        for i in random.sample(range(0, train_input.shape[0]), 5):  # 从 0 到 train_input.shape[0]-1 的范围内随机抽取5个不重复的整数索引。
            print(i)
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

            original = ax1.imshow(np.real(train_output[i, :, :]), cmap=plt.cm.jet, extent=[-0.75, 0.75, -0.75, 0.75])
            fig.colorbar(original, ax=ax1, fraction=0.046, pad=0.04)
            ax1.title.set_text("Original scatterer")

            guess_real = ax2.imshow(train_input[i, :, :, 0], cmap=plt.cm.jet, extent=[-0.75, 0.75, -0.75, 0.75])
            fig.colorbar(guess_real, ax=ax2, fraction=0.046, pad=0.04)
            ax2.title.set_text("Initial guess: real")

            guess_imag = ax3.imshow(train_input[i, :, :, 1], cmap=plt.cm.jet, extent=[-0.75, 0.75, -0.75, 0.75])
            fig.colorbar(guess_imag, ax=ax3, fraction=0.046, pad=0.04)
            ax3.title.set_text("Initial guess: imaginary")

            plt.show()

    @staticmethod
    def plot_results(gt, chi, output, save_path=None):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4)

        original = ax1.imshow(gt, cmap=plt.cm.jet, extent=[-0.75, 0.75, -0.75, 0.75])
        cb1 = fig.colorbar(original, ax=ax1, fraction=0.046, pad=0.04)
        cb1.ax.tick_params(labelsize=12)
        ax1.title.set_text(f"Original scatterer")
        ax1.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

        guess_real = ax2.imshow(chi[0], cmap=plt.cm.jet, extent=[-0.75, 0.75, -0.75, 0.75])
        cb2 = fig.colorbar(guess_real, ax=ax2, fraction=0.046, pad=0.04)
        cb2.ax.tick_params(labelsize=12)
        ax2.title.set_text("Initial guess: real")
        ax2.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

        guess_imag = ax3.imshow(chi[1], cmap=plt.cm.jet, extent=[-0.75, 0.75, -0.75, 0.75])
        cb3 = fig.colorbar(guess_imag, ax=ax3, fraction=0.046, pad=0.04)
        cb3.ax.tick_params(labelsize=12)
        ax3.title.set_text("Initial guess: imaginary")
        ax3.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

        reconstruction = ax4.imshow(output, cmap=plt.cm.jet, extent=[-0.75, 0.75, -0.75, 0.75])
        cb4 = fig.colorbar(reconstruction, ax=ax4, fraction=0.046, pad=0.04)
        cb4.ax.tick_params(labelsize=12)
        ax4.title.set_text("Reconstructed")
        ax4.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

        plt.setp(ax1.get_xticklabels(), fontsize=12, horizontalalignment="left")
        plt.setp(ax2.get_xticklabels(), fontsize=12, horizontalalignment="left")
        plt.setp(ax3.get_xticklabels(), fontsize=12, horizontalalignment="left")
        plt.setp(ax4.get_xticklabels(), fontsize=12, horizontalalignment="left")

        plt.setp(ax1.get_yticklabels(), fontsize=12)
        plt.setp(ax2.get_yticklabels(), fontsize=12)
        plt.setp(ax3.get_yticklabels(), fontsize=12)
        plt.setp(ax4.get_yticklabels(), fontsize=12)

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)
        plt.show()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {save_path}")

        plt.close(fig)

    @staticmethod
    def view_scatterer(scatterer_forward, scatterer_inverse):
        # scatterer_forward = self.generate()
        # scatterer_inverse = self.generate()

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4)

        original_real = ax1.imshow(np.real(scatterer_forward), cmap=plt.cm.jet, extent=[-0.75, 0.75, -0.75, 0.75])
        cb1 = fig.colorbar(original_real, ax=ax1, fraction=0.046, pad=0.04)
        cb1.ax.tick_params(labelsize=12)
        ax1.title.set_text("Forward scatterer: real")
        ax1.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

        guess_real = ax2.imshow(np.imag(scatterer_forward), cmap=plt.cm.jet, extent=[-0.75, 0.75, -0.75, 0.75])
        cb2 = fig.colorbar(guess_real, ax=ax2, fraction=0.046, pad=0.04)
        cb2.ax.tick_params(labelsize=12)
        ax2.title.set_text("Forward scatterer: imag")
        ax2.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

        guess_imag = ax3.imshow(np.real(scatterer_inverse), cmap=plt.cm.jet, extent=[-0.75, 0.75, -0.75, 0.75])
        cb3 = fig.colorbar(guess_imag, ax=ax3, fraction=0.046, pad=0.04)
        cb3.ax.tick_params(labelsize=12)
        ax3.title.set_text("Inverse scatterer: real")
        ax3.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

        guess_imag = ax4.imshow(np.imag(scatterer_inverse), cmap=plt.cm.jet, extent=[-0.75, 0.75, -0.75, 0.75])
        cb4 = fig.colorbar(guess_imag, ax=ax4, fraction=0.046, pad=0.04)
        cb4.ax.tick_params(labelsize=12)
        ax4.title.set_text("Inverse scatterer: imag")
        ax4.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

        plt.setp(ax1.get_xticklabels(), fontsize=12, horizontalalignment="left")
        plt.setp(ax2.get_xticklabels(), fontsize=12, horizontalalignment="left")
        plt.setp(ax3.get_xticklabels(), fontsize=12, horizontalalignment="left")
        plt.setp(ax4.get_xticklabels(), fontsize=12, horizontalalignment="left")

        plt.setp(ax1.get_yticklabels(), fontsize=12)
        plt.setp(ax2.get_yticklabels(), fontsize=12)
        plt.setp(ax3.get_yticklabels(), fontsize=12)
        plt.setp(ax4.get_yticklabels(), fontsize=12)

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)
        plt.show()


if __name__ == '__main__':

    PlotUtils.plot_setup()
