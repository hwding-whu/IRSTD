# Infrared Small Target Detection Improvement via Hybrid Data Augmentation Using Diffusion Models and GAN
## Abstract
Deep learning models based on Convolutional Neural Networks (CNNs) have been widely researched and applied in the field of Infrared Small Target Detection (IRSTD). However, current CNN-based models require a sufficient amount of training data to achieve good detection performance. The collection and annotation of infrared small targets are challenging, resulting in a quantity that cannot meet the demands of CNN-based deep learning detection models. To address this issue, we propose a two-stage infrared small target image augmentation scheme in this paper. The first stage is background generation. We first use a background filling method to obtain clean background images. Then, we generate more complex and diverse background images based on the popular deep generative model - the diffusion models. The second stage is target fusion, aiming to better integrate masks and generated background images. We designed a target adaptive fusion method based on Generative Adversarial Networks (GAN) to generate more realistic infrared small target augmented images. Experimental results on three different scene datasets show that, compared to models trained with only original data, incorporating augmented data can achieve better detection. We validated the effectiveness of the proposed method using the latest detection algorithms.
## Overall framework flowchart
![Framework diagram](https://github.com/hwding-whu/ISTD/blob/master/images/Framework%20diagram.png)
## Datasets
In this study, we utilized the NUDT-SIRST dataset as our primary dataset. You can download the relevant datasets here: [NUDT-SIRST](https://github.com/YeRen123455/Infrared-Small-Target-Detection)
## Baseline models
[U-Net & U-Net++:] (https://github.com/qubvel/segmentation_models.pytorch)

[RDIAN:] (https://github.com/sun11999/RDIAN)

[DNA-Net-18 & DNA-Net-34:] (https://github.com/YeRen123455/Infrared-Small-Target-Detection)

[UIU-Net:] (https://github.com/danfenghong/IEEE_TIP_UIU-Net)

[RPCANet:] (https://github.com/fengyiwu98/RPCANet)
