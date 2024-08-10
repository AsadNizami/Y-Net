# Y-Net: A Dual Stream Encoder-Decoder Network for Image Forgery Localization
![Y_net](https://github.com/user-attachments/assets/bc798e0b-7ec1-4973-8b3e-ac67830a47a9)
*Fig1: Overview of the Y-Net architecture*

## Table of content
- Abstract
- Impact Statement
- Models
- Training
- Results of the Ablation Study
## Abstract
As technology continues to develop at a rapid pace, new methods to deceive people are evolving as well. Image forgery has become a major concern in the digital age and poses a significant challenge in various domains. Classification of forgery is crucial for forensic analysis and verification purposes. Consequently, better techniques for forgery localization are necessary. This work introduces Y-Net, a new architecture for the forgery localization task that employs RGB and ELA (Error Level Analysis) of an image as a parallel stream of input for the two encoders. These streams are then fused and passed through a single decoder network, augmented by skip connections from the two encoder networks. Moreover, the convolutional block in the encoder and decoder module utilizes the Convolutional Block Attention Module to enhance the discriminative power of the learned features. The effectiveness of this model is demonstrated by extensive experimentation on benchmark datasets.

## Impact Statement
A tremendous amount of images are shared every day, many of them are fake and can be used for unfair means. Methods to localize forgeries in images have been developed since the last decade, however many of the methods are computationally expensive and struggle to localize the tampered region of an image in newer datasets. In this study, we developed Y-Net, a lightweight neural network model that achieved an F1 score of 68.74\%  and 82.49\% on the CASIA and the Fantastic Reality datasets, respectively. Hence, it can be useful to verify the authenticity of images because tampered images can have serious negative consequences like manipulating the general public, economic fraud and blackmailing.

## Models
- [Stage 1](https://drive.google.com/file/d/1zdU6x4aZEn6-IMasIkiBN8-X18gLSnDZ/view?usp=sharing)- Trained on over 100k images from image repositories which include CASIA, Fantastic Reality and DEFACTO.
- [Stage 2](https://drive.google.com/file/d/1gHmckhL9Y2Zy2VHsJ5EoWTzYxovNbYbT/view?usp=drive_link)- Trained on ~15000 images from the same image repositories.

## Training
```
git clone https://github.com/AsadNizami/Y-Net.git
cd Y-Net/src
```
Configure the config.py file for you environment

```
python train.py
```

## Results of the Ablation Study
![Stream Comp](https://github.com/user-attachments/assets/233e4f30-941c-4eb5-9ef2-a479b3da0a5d)
*Fig2: Contribution of different stream of input for forgery localization*

