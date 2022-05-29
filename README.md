# Few-shot Learning of CT Stroke Segmentation Based on U-Net

In order to solve the shortcomings of computer tomography and gain time for doctors to treat patients, we try to **use deep learning to help doctors diagnose ischemic stroke**. Since it is not easy to obtain images and the amount of data is small, we use few-shot learning to solve our problem.

According to the lack of brain CT image, we use several techniques to enhance the ability of segmentation like **data augmentation, pre-classification of training data by clustering**. Moreover, we've tested several segmentation model and different activation function to find the best stroke segmentation model. 

Eventually, our stroke segmentation model got **0.6384 IoU with 0.6765 sensitivity and 0.9987 specificity by using U-Net with leaky ReLU** as activation function in each layer. 

<img src=https://i.imgur.com/yBW0Uyw.jpg width=200><br>

[TOC]

<font size=6>**Dependencies**</font>
---
- python = 3.8
- tensorflow = 2.5.0
- sklearn
- scikit-image

<font size=6>**Demo**</font>
---
We observe that our data can be distinguished between few types by position, and shape. Hence, we try to use VGG16 with the pretrained weight of Imagenet to do the clustering. This can make the data is distributed in train, validation, and test dataset equally to make sure that the model has generalization enough.

- Here we divide all 56 brain CT images into 8 classes by clustering: 
![](https://i.imgur.com/saHLJz6.jpg)

- We choose the leaky ReLU U-Net as the best segmentation model, and train the model for 50 epochs with 41 brain CT images. Here is the results:
![](https://i.imgur.com/7Vhpy5p.png)

<font size=6>**Training**</font>
---
### Data preprocessing
1. Put all of your data in `data/raw/`, and give different suffix like:
    ```
    brain CT image --> 0.png
    segmentation mask --> 0_mask.png
    sythesis of CT image and mask --> 0_result.png
    ```
2. Run all cells in `Data preprocessing.ipynb`. the train and test folder will be generated automatically in `data/*`

### Start training
* You can directly run the default setting by:
    ```
    python main.py
    ````
* Or choose the model between **U-Net** and **ResU-Net** with the activation function(ResU-Net can't set activation function) and the number of epochs:
    ```
    python main.py -m ResUNet -e 50

    //or

    python main.py -m UNet -a leaky_relu -e 50
    ```

:::info
<font size=2>*\*Please notice that if you have a very small dataset (under 1000 images), you should train for several times to get the best model.*</font>
:::

<font size=6>**Evaluation**</font>
---
* Put your data in `data/test/image`, and run:
    ```
    python evaluation.py -mp model/your_stroke_segment_model.h5
    ```
    The ouput will show you some evaluation metrics like dice, sensitivity, specificity, and IoU, the calculation of IoU is shown below:
    
    <img src=https://i.imgur.com/gTREo0x.png width=300>

<font size=6>**Reference**</font>
* [zhixuhao/unet](https://github.com/zhixuhao/unet)
* [nikhilroxtomar/Deep-Residual-Unet](https://github.com/nikhilroxtomar/Deep-Residual-Unet)

###### tags: `GitHub` `Python` `ML`
