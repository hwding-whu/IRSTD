import cv2
from matplotlib import pyplot as plt
import os

img_path = 'data/128/imgs/'
mask_path = 'data/128/masks/'
img_files = os.listdir(img_path)
mask_files = os.listdir(mask_path)
print(img_files)
print(mask_files)

for i in range(len(img_files)):
    img = cv2.imread(img_path+img_files[i])
    mask = cv2.imread(mask_path+mask_files[i], 0)
    dst1 = cv2.inpaint(img,mask,3,cv2.INPAINT_NS)
    dst2 = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
    cv2.imwrite('output1/'+img_files[i], dst1)
    cv2.imwrite('output2/'+img_files[i], dst2)

