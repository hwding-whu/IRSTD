import os
import torch
import config
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2

class InfDataset(Dataset):
    def __init__(self,in_dir1,in_dir2):
        super(InfDataset, self).__init__()

        self.in_dir1 = in_dir1
        self.in_dir2 = in_dir2
        self.in_files1 = os.listdir(self.in_dir1)
        self.in_files2 = os.listdir(self.in_dir2)

    def __len__(self):
        return len(self.in_files1)

    def __getitem__(self, index):
        in_file1 = self.in_files1[index]
        in_file2 = self.in_files2[index]
        in_path1 = os.path.join(self.in_dir1, in_file1)
        in_path2 = os.path.join(self.in_dir2,in_file2)
        input_image1 = cv2.imread(in_path1, 0)
        input_image2 = cv2.imread(in_path2)
        background_mean = cv2.mean(input_image2, input_image1)[0]
        if background_mean < 38.25:
            background_mean = 51
        elif background_mean < 168.3:
            background_mean = background_mean * 1.5
        else:
            background_mean = 255
        background_filled = np.full(input_image2.shape, background_mean, dtype=np.uint8)
        result = cv2.bitwise_and(background_filled, background_filled, mask=input_image1)
        input_image = cv2.bitwise_or(result,
                                     cv2.bitwise_and(input_image2, input_image2, mask=cv2.bitwise_not(input_image1)))
        # input_image = cv2.addWeighted(input_image1, 1, input_image2, 1, 0.0)

        mask = config.mask_transforms(input_image1)
        input = config.img_transforms(input_image)


        return mask, input