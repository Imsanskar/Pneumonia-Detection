## Pneumonia-Detection

### Steps for Executing Code
* Insall Required libraries
```
python3 -m pip install -r requirements.txt
```

* Download the dataset from [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and unzip it in dataset directory.

* Download the model from [here](https://drive.google.com/drive/folders/1AL_6rWRiXy3qOmQgi0VASjNd_WN4zKMu?usp=sharing)

* To train Convolution network and VGG-19 network, run all the required cells from notebook `train_cnn.ipynb` and `train_vgg.ipynb` respectively.

* For evaluation of the models, run the notebook `model_evaluation.ipynb`

### Directory Structure
```
├── dataset
│   ├── chest_xray
│   │   ├── train
│   │   ├── test
├── models
│   ├── cnn_model.pth
│   ├── vgg_19_model.pth
├── results
├── model_evaluation.ipynb 
├── models.py       
├── train_cnn.ipynb 
├── train_vgg.ipynb 
├── utils.py
```