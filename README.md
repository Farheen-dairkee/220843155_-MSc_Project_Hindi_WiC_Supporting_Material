# Hindi_WiC

# About

## Task Description
Hindi is spoken by more than 500 million people worldwide. However, the development of NLP research in this language has been significantly slower compared to the rapid progress made in other languages. As a result, my research project is centered around tackling a specific challenge known as Word Sense Disambiguation (WSD) in Hindi. This challenge serves as a way to assess whether there truly is an issue with NLP support for the Hindi Language. To approach this challenge, I have used the Word-in-Context (WiC) [1] approach, WiC as defined on the official website (https://pilehvar.github.io/wic/) is a binary classification task. Each instance in WiC has a target word w, either a verb or a noun, for which two contexts are provided. Each of these contexts triggers a specific meaning of w. The task is to identify if the occurrences of w in the two contexts correspond to the same meaning or not. For Hindi, there are not enough helpful resources available for this method. So, I created a new dataset to help with WiC in Hindi.

This researh paper introduces a novel dataset for WiC in Hindi which is used as litmus paper test to understand the progress of NLP support in Hindi. 

In my study, I looked at different contextual embedding models and compared their performance on the WiC task. The four types used are: mBERT [2], XLM-RoBERTa [3], IndicBERT [4], and MuRIL [5]. The first one, mBERT, has been trained on 104 languages including English and Hindi. This model was used for the purpose of cross-lingual learning where the model was trained on English and tested on Hindi as well as monolingual learning where training and testing was done on Hindi Language. The second one, IndicBERT has been pretrained on Hindi corpora. This model’s performance was checked by training and testing on Hindi. The last one, MuRIL, was pretrained with 17 different Indian languages. This model was also used for the same purpose as IndicBERT.

## Notes

* There is python file that was used for: preprocessing and cleaning of target words, create the final Hindi WiC dataset as per the English WiC dataset and creating training, validation and test sets for Hindi.

* The Jupyter notebooks that are named as crosslingual and monolingual were used to run different contextual embedding models on the datasets and assess their performance. These notebooks were made to run on Google Colab.

* Apart from the Hindi WiC Dataset that has been used, the training and validation sets of the English WiC Dataset created for the SuperGLUE WiC task was also used for the purpose of transfer learning (more information is given below).

## Findings

* In the crosslingual setting, the multilingual model mBERT was trained on English and tested on Hindi, it performed slightly better than the chance level giving an accuracy of 52% and F1 score of 48%. Another multilingual model XLM-RoBERTa was used while following the same experimental setup yielded an accuracy of 55% and a similar F1 score. The performance of this model is also close to chance level. 

* In the monolingual setting, both the multilingual models mBERT and XLM-RoBERTa was trained and tested exclusively on Hindi, both performed much better than before and gave an F1 score of 79%, 83% and an accuracy of 83%, 87% respectively.

* Further experiments with the monolingual setting included training models pretrained on mainly Indian Languages that were IndicBERT and MuRIL. IndicBERT was able to give an accuracy of 62% and F1 score of 56% whereas MuRIL which has been pretrained on a larger text corpora gave an accuracy of 90% with an F1 score of 88%.

# Files

## Python Files

dataset_preprocessing.py

This code was used to read the datasets and create the hindi language train, validation and test datasets. The code reads from the original dataset "Sense Annotated Hindi Corpus" (SAHC) [6](https://ieeexplore.ieee.org/document/787592) and converts it into the WiC dataset which follows the format of the English WiC Dataset. Since the SAHC dataset can't be published without the required permissions, I have not uploaded the dataset in this repo. 

## Jupyter Notebooks

To avoid any issues in the running of the code, please run the code according to the given sequence of sections. 

crosslingual_wic.ipynb

In this notebook, I have loaded the train and validation sets of the English WiC Dataset and used the multilingual model mBERT to train on this dataset. 
The model after fine tuned on the given dataset is then tested on the Hindi WiC Dataset. 

### The code contains all the lines needed to install and import the necessary libraries.

crosslingual_wic_xlmr.ipynb

In this notebook, I have loaded the train and validation sets of the English WiC Dataset and used the multilingual model XLM-RoBERTa to train on this dataset. 
The model after fine tuned on the given dataset is then tested on the Hindi WiC Dataset. 

### The code contains all the lines needed to install and import the necessary libraries.

monolingual_wic.ipynb

In this notebook, I have loaded the created Hindi WiC Dataset and used the multilingual model mBERT, IndicBERT and MuRIL to train on this dataset. The model after fine tuned on the given dataset is then tested on the Hindi WiC Dataset. 

#### The code contains all the lines needed to install and import the necessary libraries.

Libraries Installed

```
pip install pytorch-transformers
!pip install transformers
```

Libraries Imported
```
import torch
import os
import string
import copy
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report
from pytorch_transformers import *
import numpy as np
import json
import collections
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
```


# Dataset Used

## English WiC Dataset

The format of this dataset includes target words, two sentences of same or different contexts and a label. The sentences both use the target word. The goal of this task is to classify if the target word in both sentences mean the same or not.

The dataset is available SuperGLUE benchmark site (https://super.gluebenchmark.com/tasks). The data contains 3 jsonl files, for training, validation and testing. This version of the dataset includes the character position of each target word in both sentences of each dataset entry. This character position helps to create mask tensors in my version of experiment. Also, for the purpose of my experiment test set is not used.

## Hindi WiC Dataset

The format of this dataset includes target words, two sentences of same or different contexts and labels namely 0 (False) and 1 (True). These sentences contain the target word and could denote the same meaning or not. This dataset also contains charachter positions of the target words in the sentences. The training set contains 7000 instances, validation set contains 1000 instances and the test set contains 2000 instances.

This dataset was created with the help of "Sense Annotated Hindi Corpus" whose paper can be found here, https://ieeexplore.ieee.org/document/7875926.  

# References
