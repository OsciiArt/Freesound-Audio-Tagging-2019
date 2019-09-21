# Freesound-Audio-Tagging-2019

This is repository of the 4th place solution of
 [kaggleFreesound Audio Tagging 2019 competition](https://www.kaggle.com/c/freesound-audio-tagging-2019/).  
The discription of this solution is available at  
http://dcase.community/challenge2019/task-audio-tagging-results#Akiyama2019  
https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion/96440

# Requirements
- Python 3.6.6
- CUDA 10.0
- numpy (1.16.4)
- pandas (0.23.4)
- matplotlib (3.1.0)
- Pytorch (1.1.0)
- librosa (0.6.3)
- sci-kit learn (0.21.2)
- scipy (1.2.1)
- pretrainedmodels (0.7.4)

Download the [dataset](https://www.kaggle.com/c/freesound-audio-tagging-2019/data) 
and place them in `input/`.  
Unzip zip files and place them to `train_curated/`, `train_noisy/`, `test/`.   
In case you use pretrained weights, download the [weights](https://www.kaggle.com/osciiart/freesound2019-pretrained-weight-links), 
unzip zipped weights and place them to `models/resnet_model1/`, `models/resnet_model2/`  and so on.  

# Training
Run `src/preprocess.py`.  
Run `src/train_model1.py`.   
Run `src/get_pseudo_label.py`.   
Run `src/train_model2.py` .  
Run `src/train_model3.py` .  
Run `src/train_model4_0.py`.  
Run `src/train_model4.py`.  
Run `src/train_model5.py`.  
Run `src/train_model6_0.py`.  
Run `src/train_model6.py`.  

# Prediction
Run `src/make_final_submission1.py`. The submission file `output/submission1.csv` will be generted.   
Run `src/make_final_submission2.py`. . The submission file `output/submission2.csv` will be generted.      



