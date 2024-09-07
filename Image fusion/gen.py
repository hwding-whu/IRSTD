from net import Generator_128
import config
import torch
from dataset_gen import InfDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import cv2

net_G = Generator_128.Generator(in_channles=3).to(config.DEVICE)
net_G.load_state_dict(torch.load(config.load_model, map_location=config.DEVICE)['net_G'])

val_dataset = InfDataset(config.GEN_masks, config.GEN_imgs)
val_loader = DataLoader(dataset=val_dataset,batch_size=len(val_dataset),shuffle=False)

mask, x = next(iter(val_loader))
mask, x = mask.to(config.DEVICE), x.to(config.DEVICE)

counter=0
with torch.no_grad():
    y_fake = net_G(x).cpu().permute(0, 2, 3, 1).numpy()
    y_fake = (y_fake + 1) * 0.5 * 255
    for image in y_fake:
        fake = image.reshape(128, 128, 3)
        im_gray = cv2.cvtColor(fake, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(config.gen_folder+'/gen_test', '1-1-%d.png' % counter), im_gray)
        save_image(x[counter] * 0.5 + 0.5, os.path.join(config.gen_folder+'/input_test', '1-1-%d.png' % counter))
        save_image(mask[counter], os.path.join(config.gen_folder+'/masks_test', '1-1-%d.png' % counter))
        counter += 1