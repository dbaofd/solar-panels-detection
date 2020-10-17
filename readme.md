### Automatic Detection of Solar Panels on High Resolution Imagery.
# Involved Models 
- SegNet (https://ieeexplore.ieee.org/abstract/document/7803544)
- Fast SCNN (https://arxiv.org/pdf/1902.04502.pdf)
- ResNet (https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
# Code Base
- SegNet (https://github.com/advaitsave/Multiclass-Semantic-Segmentation-CamVid/blob/master/Multiclass_Semantic_Segmentation_using_VGG_16_SegNet.ipynb)
- Fast SCNN (https://github.com/kshitizrimal/Fast-SCNN/blob/master/TF_2_0_Fast_SCNN.ipynb)
# Trained Models 
- SegNet 1: SegNet with 4 encoders and 4 decoders.
- SegNet 2: SegNet with 5 encoders and 5 decoders, each encoder is replaced by a ResNet block (Block with 3 convolutional layers).
- SegNet 3: SegNet with 5 encoders and 5 decoders, each encoder is replaced by a ResNet block (Block with dynamic number of convolutional layers).
- Fast SCNN 1: Fast SCNN with the first two DSConv layers removed, modified Upsample layers.
- Fast SCNN 2: Fast SCNN with the first two DSConv layers replaced with Conv layers, modified Upsample layers. 
# Evaluations
|Model Name|Evaluation IoU| 
|---|---|
|SegNet 1|0.8196788|
|SegNet 2|0.78407985|
|SegNet 3|0.7865806|
|Fast SCNN 1|0.72767824|
|Fast SCNN 2|0.82243156|

