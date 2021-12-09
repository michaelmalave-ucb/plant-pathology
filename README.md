# Students: Mili Gera, Michael Malavé, Jonathan Miller, Heather Rancic
## W207 Final Project: Apple Disease Detection
=======
#### Related documentation: [Plant Pathology Doc](https://docs.google.com/document/d/1RpNuKjtVvOmK_hWAYhUnj_zdjdlhGfQP4Lc0vnMacME/edit?usp=sharing)

### Problem Statement
> * The US apple industry alone is worth 15 billion dollars. Unfortunately even one apple disease can causes losses up to 70%. The key to avoiding these losses is early disease detection. The current process is highly manual where trained disease scouts personally scan orchards to look for symptoms. It is not possible for disease experts to cover entire orchards and nor is it possible to easily train new disease experts.

> * We would like to explore if it is possible to use machine learning models to detect disease in a way similar to that used by disease scouts. That is, can machine learning models detect apple disease by "looking" at symptoms of disease on the images of apple leaves?

### Data
> * Data can be obtained from the following kaggle competition: https://www.kaggle.com/c/plant-pathology-2021-fgvc8/data
> * This project uses unzipped data from kaggle manually added to the data file. 
> * You should expect test_images and train_images directories and train.csv file.
> * See .gitignore for ignored files.

### Table of Contents
> * preprocessing_mg_v2.py: this python script does some pre-processing (such as sizing, etc.) to the images to prepare them for various machine learning algorithms.
> * project_mg_v2.py: this python script will run and produce the results of various KNN models.
> * project_mg_logreg.py: this python script will run and produce the results of various Logistic Regression models.
> * image_cnn.py: ...
> * README.md: this is the current document which describes the W207 final project.

### Tools Used
> * We used regular python scripts along with the classification model packages available through scikit-learn and scikit-multilearn.

### How To Run The Files
> * Download data via the kaggle link provided into a folder of your choice.
> * Once downloaded, create and move the file contents to a subfolder called "data".
> * In the data folder you should now see a subfolder called "train_images" which will contain the images of the apple leaves. In addition, there should be a file called "train.csv" which maps the image file names to the target variables (i.e. the diseases detected on the image leaves).
> * To run the KNN and Logistic Regression models, first run the file called "preprocessing_mg_v2.py". 
> * To run the KNN models, please run the file called project_mg_v2.py.
> * To run the Logistic Regression models, please run the file called project_mg_logreg.py.
