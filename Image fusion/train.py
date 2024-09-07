import torch
import config
import numpy as np
from tqdm import tqdm
from dataset import InfDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from net import Generator_128,Discriminator,Dis_patch
from torchvision.utils import save_image
from utils import save_checkpoint,save_some_examples,load_checkpoint,Hinge_loss
from torchsummary import summary

def train_fn(disc,gen,train_loader,opt_disc,opt_gen,MSE,L1_LOSS):
    loop = tqdm(train_loader,leave=True)
    for idx,(x1,x2,x,y) in enumerate(loop):
        x1,x2,x,y = x1.to(config.DEVICE),x2.to(config.DEVICE),x.to(config.DEVICE),y.to(config.DEVICE)
        # print(x.size(),y.size())

        #train dsicriminator
        y_fake = gen(x)
        D_real = disc(y)
        D_fake = disc(y_fake.detach())
        D_real_loss = MSE(D_real,torch.ones_like(D_real))
        D_fake_loss = MSE(D_fake,torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss)/2
        # D_loss = Hinge(D_real, D_fake)

        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        #train generator
        # with torch.cuda.amp.autocast():
        D_fake = disc(y_fake)
        G_fake_loss = MSE(D_fake,torch.ones_like(D_fake))
        # G_fake_loss = Hinge(D_fake)
        L1 = L1_LOSS(y_fake,y)*config.L1_LAMBDA
        with torch.no_grad():
            BM = x2.mul(x1)
            FM = y_fake.mul(x1)
        # LF = torch.abs(BM-FM).mean() * config.LF_LAMBDA
        LF = L1_LOSS(FM,BM)*config.LF_LAMBDA
        G_loss = G_fake_loss + L1 + LF
        # print(G_fake_loss, L1, LF)

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )



def main(i):
    # disc = Discriminator_patch.Discriminator_patch().to(config.DEVICE)
    IN_DIR_1 = "data/128_new/%d/train/masks"%i
    IN_DIR_2 = "data/128_new/%d/train/imgs"%i
    TRA_DIR = "data/128_new/%d/train/labels"%i

    TIN_DIR_1 = "data/128_new/%d/train/masks"%i
    TIN_DIR_2 = "data/128_new/%d/train/imgs"%i
    TTRA_DIR = "data/128_new/%d/train/labels"%i

    save_model = 'models_batch/%d/model.pt'%i
    sample_folder = "results_batch/%d"%i

    disc_patch = Dis_patch.PatchDiscriminator().to(config.DEVICE)
    gen = Generator_128.Generator(in_channles=3).to(config.DEVICE)


    opt_disc_patch = torch.optim.Adam(disc_patch.parameters(),lr = config.LEARNING_RATE,betas=(0.5,0.999))
    opt_gen = torch.optim.Adam(gen.parameters(),lr = config.LEARNING_RATE,betas=(0.5,0.999))

    MSE = torch.nn.MSELoss()
    # BCE_LOSS = torch.nn.BCELoss()
    L1_LOSS = torch.nn.L1Loss()

    train_dataset = InfDataset(IN_DIR_1, IN_DIR_2, TRA_DIR)
    val_dataset = InfDataset(TIN_DIR_1, TIN_DIR_2, TTRA_DIR)

    train_loader = DataLoader(dataset=train_dataset,batch_size=config.BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,batch_size=1,shuffle=True)

    # g_scaler = torch.cuda.amp.GradScaler()
    # d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        # train_fn(disc,gen,train_loader,opt_disc,opt_gen,L1_LOSS,BCE,g_scaler,d_scaler)
        train_fn(disc_patch, gen, train_loader, opt_disc_patch, opt_gen, MSE, L1_LOSS)

        if config.SAVE_MODEL and epoch % 20 == 0:
            torch.save({
                'net_G': gen.state_dict(),
                'net_D': disc_patch.state_dict(),
                'net_D_patch': disc_patch.state_dict()
            }, save_model)
        save_some_examples(gen,val_loader,epoch,folder=sample_folder)


if __name__ == '__main__':
    for i in range(1,3):
        main(i)