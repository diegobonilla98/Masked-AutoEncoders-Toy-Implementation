import glob
import os
import torch.nn
import numpy as np
import tqdm
from timm.optim import optim_factory
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision
from PIL import Image
from model import mae_vit_large_patch16 as Model
from util import NativeScalerWithGradNormCount as NativeScaler
from util import adjust_learning_rate


class CustomDataset(Dataset):
    def __init__(self, size):
        self.images = glob.glob("D:\\102flowers\\train\\**\\*")
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


BATCH_SIZE = 80
LEARNING_RATE = 1.5e-4 * BATCH_SIZE / 256
USE_CUDA = torch.cuda.is_available()
N_EPOCHS = 100
IMAGE_SIZE = (224, 224)

model = Model()
print(model)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f"Num. Printable Parameters: {params:,}")

data_loader = DataLoader(CustomDataset(IMAGE_SIZE), batch_size=BATCH_SIZE, shuffle=True)

param_groups = optim_factory.add_weight_decay(model, 0.05)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))
loss_scaler = NativeScaler()

if USE_CUDA:
    model = model.cuda()

for p in model.parameters():
    p.requires_grad = True

best_loss = np.inf
epochs_without_improvement = 0

for epoch in range(N_EPOCHS + 1):
    data_iter = iter(data_loader)
    i = 0
    epoch_losses = []
    with tqdm.tqdm(total=len(data_loader)) as pbar:
        while i < len(data_loader):
            sample = next(data_iter)
            image, label = sample

            adjust_learning_rate(optimizer=optimizer, epoch=i / len(data_loader) + epoch, warmup_epochs=N_EPOCHS // 10, lr=LEARNING_RATE, min_lr=0, epochs=N_EPOCHS)

            optimizer.zero_grad()
            if USE_CUDA:
                image = image.cuda()
                label = label.cuda()

            loss, _, _, _ = model(image)

            loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(i + 1) % 1 == 0)

            i += 1

            epoch_losses.append(loss.item())

            pbar.set_description(f"Iter: {i}/{len(data_loader)}, [Loss: {epoch_losses[-1]}]")
            pbar.update()

    epoch_loss = np.mean(epoch_losses)
    if best_loss - epoch_loss > 0.05:
        epochs_without_improvement = 0
        best_loss = epoch_loss
        torch.save(model, f'./checkpoints/best_model.pth')
    else:
        epochs_without_improvement += 1
        # scheduler.step()
        if epochs_without_improvement == 5:
            break
    print(f'[Epoch: {epoch}/{N_EPOCHS}, [Loss: {epoch_loss}]')

