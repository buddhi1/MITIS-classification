# A Survey on MITIS Image Classification Using Different Artificial Intelligence Models

**Group Members**: Desiree Garcia, Buddhi Ashan Mallika Kankanamalage, Heila Shahidi, Paul Woody, Mark Zimmerschied

## Posterboard Presentation
xxxxxxxx

## Posterboard Link
https://tinyurl.com/yckhrepm

## Abstract
Scene classification is a challenging task in computer vision due to ambiguity, variability, scale variations, and different illumination conditions. Also, significant differences between the same class images makes this problem harder. To reduce complexity of this problem, sub categorized datasets have been published. This work focuses on indoor scene classification using MIT Indoor Scene (MITIS) dataset. Recent advancement in Deep Neural Networks (DNN) have enhanced the performance of scene classification even though its performance is still low compared to some other image classification problems. This study encapsulates a comparison of state-of-the-art algorithms for scene classification. Our experiments show that ResNetXt-101 feature descriptor-based Support Vector Machine (SVM) outperforms other selected models with 76% accuracy. ​

## Introduction
Scene classification is the task of labeling an image based on a predefined category. This work focuses on indoor scene classification using the MITIS dataset which consists of 15,620 images categorized into 67 indoor categories with each category containing at least 100 images. Other noteworthy datasets are Places205 with 205 scene categories and 2.5 millions images, and Places365-Standard with 365 scene categories and 1.8 million images.​

Boundaries between some scenes are hard to define and some objects can appear in multiple scene categories. Therefore, this task requires identifying distinctive characteristics and similarities between same class images [3].  DNN approaches such as Convolutional Neural Networks (CNN) have proved this kind of behavior [3, 4]. In this work, we compare five Artificial Intelligence (AI) models using different evaluation matrices. 
![Sample](https://user-images.githubusercontent.com/45516923/165963763-ddd23c1f-a4b6-4d9f-9db6-a9241e9f8d44.PNG)

## Purpose
Objective of this work is to study state-of-the-art algorithms which can be used to classify the MITIS dataset. We chose five models with highest classification accuracies where the training and testing codes were publicly available. In this work we present a comparison of the classification models using different evaluation matrices. 

## Methodology
The MITIS dataset consists of 5360 train images and 1340 test images. We use the same split as published in [1]. Following are the five models used in this study.

* Convolutional Neural Network (CNN)[2]
* Support Vector Machine (SVM)[3]
* Artificial Neural Network (ANN) [3]
* Random Forest (RF)[3]
* Naive Bayes[3]

The traditional models uses feature descriptors extracted from RESNetXt-101, which is a well-known CNN. The CNN model is a transfer learning based Xception model and fine-tuned to improve efficiency. Comparison is based on accuracy, precision, recall, and F1-score matrices. Since this is a multi-class classification, we used weighted and macro approaches. 

## Results
![Weighted](https://user-images.githubusercontent.com/45516923/165962363-239b8b64-7f99-4d3a-b951-c1eb67e038ea.PNG)

![Marco](https://user-images.githubusercontent.com/45516923/165961896-347a7b6d-b197-45a7-befd-b141adbad107.PNG)

![Matrix](https://user-images.githubusercontent.com/45516923/165963298-26ade3fc-6cc0-4978-8f69-95a1a50f2d1e.PNG)

## Summary
* SVM outperforms all models with highest evaluation matrices in both weighted and macro​
* Fastest model to train is Naive Bayes​
* SVM support is 1338​
* SVM training time is relatively faster. But feature extraction can be a bottleneck  


## How to Run the Models
- Download MITIS dataset from <a href="https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019">here</a>. Unzip the downloaded file and locate it outside master directory. 
- Create anaconda environment with the given yml file using 
```
conda env create -f environment.yml
```
- Download Feature file from <a href="https://drive.google.com/file/d/1Yt8boWVIr_WHCqlRB0BtePqysdtwwYOw/view?usp=sharing">here</a> and place it in the Dataset directory.
- To avoid training the CNN, download weight file from <a href="https://drive.google.com/file/d/1uAGiIaYTBrvoHPY0SNnCDaanjiZEk9u3/view?usp=sharing">here</a> and place it in models/cnn directory
- To train all models and generate evaluation matrices, use 
```
python  python train_models.py all
```
- To train only selected models, use 
```
python train_models.py {randomforest, naivebayes, svm, nn, cnn}
```
- To only evaluate CNN, use 
```
python train_models.py cnn load
```


## Experimental Setup
Our experimental setup was a Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz equipped workstation with Nvidia Quadro RTX 5000 GPU card with 64GB of memory. 
We compared the models using accuracy, precision, recall, and F1-score for both weighted and macro. 


## Acknowledgements 
We would like to acknowledge contributions from Gamage [3] and Rahimzadeh at al. [2] as well as making their implementations publicly available through git hub. 


## References
[1] A. Quattoni, and A.Torralba. Recognizing Indoor Scenes. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2009.<br />
[2] Rahimzadeh, Mohammad, et al. "Wise-srnet: A novel architecture for enhancing image classification by learning spatial resolution of feature maps." arXiv preprint arXiv:2104.12294 (2021).<br />
[3] Gamage, Bhanuka Manesha Samarasekara Vitharana. "An embarrassingly simple comparison of machine learning algorithms for indoor scene classification." arXiv preprint arXiv:2109.12261 (2021).

***
