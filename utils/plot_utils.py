import matplotlib.pyplot as plt
import torch


def save_multiple_images(im_list, fname):
    fig, ax = plt.subplots(1, len(im_list), squeeze=False)
    for i, im in enumerate(im_list):
        ax[0, i].imshow(im.detach().cpu().numpy())

    plt.savefig(fname)
    plt.close(fig)