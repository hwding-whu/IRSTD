
import torch
import config
from torchvision.utils import save_image
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

def save_some_examples(gen, val_loader, epoch, folder):
    x1, x2, x, y = next(iter(val_loader))
    x1, x2, x, y = x1.to(config.DEVICE), x2.to(config.DEVICE), x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x).cpu().permute(0, 2, 3, 1).numpy()
        y_fake = (y_fake+1)*0.5*255
        y_fake = y_fake.reshape(128, 128, 3)
        im_gray = cv2.cvtColor(y_fake, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(folder + f"/y_gen_{epoch}.png", im_gray)
        # y_fake = gen(x)
        # save_image(y_fake * 0.5 + 0.5, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
        # if epoch == 1:
        #     save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def Hinge_loss(pred_real, pred_fake=None):
    if pred_fake is not None:
        loss_real = F.relu(1 - pred_real).mean()
        loss_fake = F.relu(1 + pred_fake).mean()
        return loss_real + loss_fake
    else:
        loss = -pred_real.mean()
        return loss