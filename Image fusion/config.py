import torch
import albumentations
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
##################Training#################
IN_DIR_1 = "data/128_new/1/train/masks"
IN_DIR_2 = "data/128_new/1/train/imgs"
TRA_DIR = "data/128_new/1/train/labels"

TIN_DIR_1 = "data/128_new/1/test/masks"
TIN_DIR_2 = "data/128_new/1/test/imgs"
TTRA_DIR = "data/128_new/1/test/labels"

save_model = 'models_batch/1/model.pt'
sample_folder = "results_batch/1"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 0
IMAGE_SIZE = 128
CHANNELS_IMG = 3
L1_LAMBDA = 100
LF_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 1000
LOAD_MODEL = False
SAVE_MODEL = True

##################Test#################
load_model = 'models_batch/1/model.pt'
GEN_masks = "data/gen/1/masks"
GEN_imgs = "data/gen/1/DF1"
gen_folder = 'gen_images/1'

transform = albumentations.Compose(
    [
        albumentations.Resize(width=128, height=128),
        albumentations.Normalize(mean=0.5, std=0.5, max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# mask只需要转换为tensor
mask_transforms = transforms.ToTensor()