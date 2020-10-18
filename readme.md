# Automatic Detection of Solar Panels on High Resolution Imagery.
## Involved Models 
- SegNet (https://ieeexplore.ieee.org/abstract/document/7803544)
- Fast SCNN (https://arxiv.org/pdf/1902.04502.pdf)
- ResNet (https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
## Dataset
- Dataset are made with high-resolution satellite imagery from Nearmap (https://www.nearmap.com/au/en)
- Labelme (https://github.com/wkentaro/labelme) is used as the tool to label the images.
- Training dataset contains 3936 256x256 rgb images and labels which are collected
from Capalaba, Springfield, New Farm, Fairfield, Sunnybank Hills in Brisbane, Australia.
- Validation dataset contains 1344 images and labels which are collected from
Springfield and Sunnybank Hills.
- Test dataset contains 1360 images and labels which are collected from a suburb in Perth, Australia.
## Trained Models 
- SegNet 1: SegNet with 4 encoders and 4 decoders.
- SegNet 2: SegNet with 5 encoders and 5 decoders, each encoder is replaced by a ResNet block (Block with 3 convolutional layers).
- SegNet 3: SegNet with 5 encoders and 5 decoders, each encoder is replaced by a ResNet block (Block with dynamic number of convolutional layers).
- Fast SCNN 1: Fast SCNN with the first two DSConv layers removed, modified Upsample layers.
- Fast SCNN 2: Fast SCNN with the first two DSConv layers replaced with Conv layers, modified Upsample layers. 
## Evaluations
|Model Name|Evaluation IoU| 
|---|---|
|SegNet 1|0.8196788|
|SegNet 2|0.78407985|
|SegNet 3|0.7865806|
|Fast SCNN 1|0.72767824|
|Fast SCNN 2|0.82243156|

