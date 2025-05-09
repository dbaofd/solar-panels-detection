# Automatic Detection of Solar Panels in High-Resolution Aerial Imagery
![profile](/imgs/cover.png)
## :book: Contents
<!--ts-->
   * [Introduction](#running-introduction)
   * [Requirements](#zap-requirements)
   * [Involved Models](#bread-involved-models-)
   * [Dataset](#dog-dataset)
   * [Trained Models](#cat-trained-models-)
   * [Evaluations](#koala-evaluations)
   * [Run Solar Panel Detection](#run-solar-panel-detection)
   * [Demo](#demo)
<!--te-->
## :running: Introduction
The global shift toward renewable energy has led to a rapid increase in the deployment of solar panels across residential, commercial, and industrial areas. 
Accurately mapping the location and extent of these installations is crucial for energy infrastructure planning, policy-making, and environmental monitoring.
With the growing availability of high-resolution aerial imagery and advances in computer vision, automated detection methods offer a scalable and cost-effective 
alternative. This project aims to develop a system that automatically detects solar panels in aerial images using state-of-the-art computer vision techniques.
## :zap: Requirements
```commandline
tensorflow = 2.16.2
opencv = 4.11.0
matplotlib = 3.10.1
```
## :bread: Involved Models 
- [SegNet](https://ieeexplore.ieee.org/abstract/document/7803544)
- [Fast SCNN](https://arxiv.org/pdf/1902.04502.pdf)

## :dog: Dataset
- Dataset are made with high-resolution aerial imagery from [Nearmap](https://www.nearmap.com/au/en)
- [Labelme](https://github.com/wkentaro/labelme) is used as the tool to label the images.
- Training dataset contains 3936 256x256 rgb images and labels which are collected
from Capalaba, Springfield, New Farm, Fairfield, Sunnybank Hills in Brisbane, Australia.
- Validation dataset contains 1344 images and labels which are collected from
Springfield and Sunnybank Hills.
- Test dataset contains 1360 images and labels which are collected from a suburb in Perth, Australia.
- <ins>Dataset will be made available in the near future.</ins>

## :cat: Trained Models 
- SegNet 0: SegNet with 5 encoders and 5 decoders (Original SegNet).
- SegNet 1: SegNet with 4 encoders and 4 decoders.
- SegNet 2: SegNet with 5 encoders and 5 decoders, each encoder is replaced by a ResNet block (Block with 3 convolutional layers).
- SegNet 3: SegNet with 5 encoders and 5 decoders, each encoder is replaced by a ResNet block (Block with dynamic number of convolutional layers).
- Fast SCNN 0: Original Fast SCNN.
- Fast SCNN 1: Fast SCNN with the first two DSConv layers removed, modified Upsample layers.
- Fast SCNN 2: Fast SCNN with the first two DSConv layers replaced with Conv layers, modified Upsample layers. 

## :koala: Evaluations
|Model Name|Evaluation IoU| 
|---|---|
|SegNet 0|0.8047909|
|SegNet 1|0.8196788|
|SegNet 2|0.78407985|
|SegNet 3|0.7865806|
|Fast SCNN 0|0.6196553|
|Fast SCNN 1|0.72767824|
|Fast SCNN 2|0.82243156|

## Run Solar Panel Detection
- Create "test_images" under the root project folder. Prepare high-resolution aerial/satellite images with arbitrary size.
Place these images inside "test_images" folder. A test image example can be downloaded [here](https://drive.google.com/file/d/1RCcGNtCsIPqC-J-6Qe6SD7V4BxpLbndN/view?usp=sharing).
- Create "prediction_images" under the root project folder. The prediction of the model will be saved here.
- Modify "solar_panel_detection.py". In line 16, modify "IMAGE_LIST", put test image names inside.
- Modify "solar_panel_detection.py". In line 118, modify "model_type" (segmentation model), "model_name" (model weights name), "test_image_name", and "saving_image_name" .
- Run the following command to perform solar panel detection on the given test image.
```commandline
python solar_panel_detection.py 
```

## Demo
Demo 1
<div>
  <img width="100%" alt="demo 1" src="imgs/demo_1.gif">
</div>
Demo 2
<div>
  <img width="100%" alt="demo 2" src="imgs/demo_2.gif">
</div>