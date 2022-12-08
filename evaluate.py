import torch
from torchvision import transforms
from torch.utils.data import Dataset
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


class CustomDataset(Dataset):
    def __init__(self, size):
        self.images = glob.glob("D:\\102flowers\\valid\\**\\*")
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = self.transform(image)
        label = torch.from_numpy(np.array(int(self.images[idx].split(os.sep)[-2]))).type(torch.LongTensor)
        return image, label


model = torch.load('checkpoints_base\\best_model.pth')
model.eval()

dataset = CustomDataset((224, 224))

with torch.no_grad():
    for idx, (image, label) in enumerate(dataset):
        image = image.cuda().unsqueeze(0)
        _, pred, mask, latent = model(image)

        pred = model.unpatchify(pred)
        pred = torch.einsum('nchw->nhwc', pred).detach().cpu()

        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)
        mask = model.unpatchify(mask)
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        image = torch.einsum('nchw->nhwc', image).detach().cpu()

        im_masked = image * (1 - mask)
        im_paste = image * (1 - mask) + pred * mask

        plt.rcParams['figure.figsize'] = [24, 24]
        plt.subplot(1, 4, 1)
        show_image(image[0], "original")
        plt.subplot(1, 4, 2)
        show_image(im_masked[0], "masked")
        plt.subplot(1, 4, 3)
        show_image(pred[0], "reconstruction")
        plt.subplot(1, 4, 4)
        show_image(im_paste[0], "reconstruction + visible")
        plt.show()
        plt.close()
