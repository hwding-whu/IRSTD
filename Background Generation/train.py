from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torchinfo import summary

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)
summary(model)
diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)
summary(diffusion, input_size=(1,3,128,128))
trainer = Trainer(
    diffusion,
    folder='data/128/0',
    results_folder = './results/0',
    save_and_sample_every = 5000,
    train_batch_size = 16,
    train_lr = 8e-5,
    train_num_steps = 50000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False              # whether to calculate fid during training
)

# trainer.train()