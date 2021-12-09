#W207 Final Project: Apple Disease Detection
### Participants: Mili Gera, Michael Malavé, Jonathan Miller, Heather Rancic

### Related documentation: [Plant Pathology Doc](https://docs.google.com/document/d/1RpNuKjtVvOmK_hWAYhUnj_zdjdlhGfQP4Lc0vnMacME/edit?usp=sharing)

### Table of Contents
> * [preprocessing_mg_v2.py](preprocessing_mg_v2.py): handles pre-processing (such as sizing, etc.) of the images preparing them for various machine learning algorithms.
> * [project_mg_v2.py](project_mg_v2.py): produces the results of various KNN models.
> * [project_mg_logreg.py](project_mg_logreg.py): produces the results of various Logistic Regression models.
> * [image_cnn.py](image_cnn.py): produces a neural network fit for multi-label classification.

## Problem Statement
Can machine learning models detect apple disease by "looking" at symptoms of disease on the images of apple leaves?
 - The US apple industry alone is worth 15 billion dollars. Unfortunately even one apple disease can causes losses up to 70%. The key to avoiding these losses is early disease detection. The current process is highly manual where trained disease scouts personally scan orchards to look for symptoms. It is not possible for disease experts to cover entire orchards and nor is it possible to easily train new disease experts.
 - We would like to explore if it is possible to use machine learning models to detect disease in a way similar to that used by disease scouts.

## Data
 - Data can be obtained from the following kaggle competition: https://www.kaggle.com/c/plant-pathology-2021-fgvc8/data
 - This project uses unzipped data from kaggle manually added to the data file. 
 - You should expect test_images and train_images directories and train.csv file.
 - See .gitignore for ignored files.

## Preparing Data To Run The Files
 - Download data via the kaggle link provided into a folder of your choice.
 - Once downloaded, create and move the file contents to a subfolder called "data".
 - In the data folder you should now see a subfolder called "train_images" which will contain the images of the apple leaves. In addition, there should be a file called "train.csv" which maps the image file names to the target variables (i.e. the diseases detected on the image leaves).

