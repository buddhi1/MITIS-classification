# A Survey on MITIS Image Classification Using Different Artificial Intelligence Models

**Group Members**: Desiree Garcia, Buddhi Ashan Mallika Kankanamalage, Heila Shahidi, Paul Woody, Mark Zimmerschied

## Problem Statement

MIT Indoor Scenes (MITIS)[1] is an image dataset with 67 indoor categories. We propose to compare five Artificial Intelligence (AI) methodologies to predict the indoor category of the MITIS images and evaluate the quality of predictions using different evaluation matrices. To improve prediction quality in traditional classification models, we propose to use Deep Learning Neural Network (DNN) based feature extractors. In this work, we present a comparison of different AI models based on their classification accuracies on MITIS dataset.

We propose following well known models to classify the MITIS dataset. 
* Convolutional Neural Network (CNN)[2]
* Support Vector Machine (SVM)[3]
* Artificial Neural Network (ANN) [3]
* Random Forest (RF)[3]
* Naive Bayes[3]

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
Our experimental setup was a Xeon silver equipped workstation with Nvidia Quadro A500 GPU card with 64 GB of memory. 
We propose to use a confusion matrix to evaluate each model.

## Acknowledgements 
We would like to acknowledge contributions from Gamage [3] and Rahimzadeh at al. [2] as well as making their implementations publicly available through git hub. 


## References
[1] A. Quattoni, and A.Torralba. Recognizing Indoor Scenes. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2009.<br />
[2] Rahimzadeh, Mohammad, et al. "Wise-srnet: A novel architecture for enhancing image classification by learning spatial resolution of feature maps." arXiv preprint arXiv:2104.12294 (2021).<br />
[3] Gamage, Bhanuka Manesha Samarasekara Vitharana. "An embarrassingly simple comparison of machine learning algorithms for indoor scene classification." arXiv preprint arXiv:2109.12261 (2021).

***