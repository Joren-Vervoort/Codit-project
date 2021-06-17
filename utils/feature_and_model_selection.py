# Imports

# Standard libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Handeling .wav files

import librosa
from librosa import feature

# Machine Learning

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.feature_selection import RFE

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from functools import reduce

# data vizualisation

import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns

# saving & loading ML method
import pickle

def concatenate_pd(machine):

    """
    Function create a merged .csv file out of all the .csv files (at every dB level)
    for a certain machine (fan, slider, pump or valve)
    : attrib machine
    This function will return a .csv file of all sound features of a certain machine
    """
    
    # Opening the .csv files
    df_6dB=pd.read_csv(f'.\\data\\created_csv_files\\Librosa_features_{machine}_6dB.csv')
    df_0dB=pd.read_csv(f'.\\data\\created_csv_files\\Librosa_features_{machine}_0dB.csv')
    df_min6dB=pd.read_csv(f'.\\data\\created_csv_files\\Librosa_features_{machine}_-6dB.csv')
    
    # Dropping the 'Unnamed: 0' column
    df_6dB.drop(columns = ['Unnamed: 0'], axis=1, inplace=True)
    df_0dB.drop(columns = ['Unnamed: 0'], axis=1, inplace=True)
    df_min6dB.drop(columns = ['Unnamed: 0'], axis=1, inplace=True)
    
    # Merging the .csv files into one DataFrame
    data_frames = [df_6dB, df_0dB, df_min6dB]
    df_merged = pd.concat(data_frames)
    
    
    return df_merged

def undersampling(df_merged): 

    """
    Function is used to balance the dataset by undersampling for a certain machine
    (fan, slider, pump or valve)
    : attrib machine
    This function will return the undersampled DataFrame 
    """

    minority_class_len = len(df_merged[df_merged['normal(0)/abnormal(1)'] == 1])
    majority_class_indices = df_merged[df_merged['normal(0)/abnormal(1)'] == 0].index
    random_majority_indices = np.random.choice(majority_class_indices, minority_class_len , replace= False)
    minority_class_indices  = df_merged[df_merged['normal(0)/abnormal(1)'] == 1].index

    under_sample_indices = np.concatenate( [minority_class_indices , random_majority_indices])
    under_sampled_data = df_merged.loc[under_sample_indices]

    return under_sampled_data

def split_data(under_sampled_data):

    """
    Function that is used to balance the dataset by undersampling for a certain machine
    (fan, slider, pump or valve)
    : attrib machine
    This function will return the undersampled DataFrame 
    """
      
    X = under_sampled_data.drop(columns = ['normal(0)/abnormal(1)'])
    y = under_sampled_data['normal(0)/abnormal(1)']

    # 30% of the overal data will seperated for later validation of the model
    X_model, X_valid, y_model, y_valid = train_test_split(X, y, test_size=0.3, random_state = 42, stratify = y)

    # 46,7% of the overal data (70% of X_model, y_model) will be used to create a training set for the model
    # 23,3% of the overal data (30% of X_model, y_model) will be used to create a testing set for the model
    X_train, X_test, y_train, y_test = train_test_split(X_model,
                                                       y_model,
                                                       test_size = 1/3,
                                                       random_state = 10, stratify=y_model)

    return X_train, X_test, X_valid, y_train, y_test, y_valid

def model_selection(X_train, X_test, X_valid, y_train, y_test, y_valid):

    """
    Function that is used to evaluate predictions of ML model used to predict the normal/abnormal
    soundpatterns for a certain machine (fan, slider, pump or valve) and save these models
    : attrib X_train, X_test, X_valid, y_train, y_test, y_valid
    This function will print a custom made report to evaluate the models performance and
    save every model with each a certain amount of features between 1 and 8
    """ 

    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', SVC())])

    y_pred = []
    clfs = []

    # append classifiers to pipeline to test their performance
    clfs.append(DecisionTreeClassifier())
    clfs.append(RandomForestClassifier())

    # generate custom made report of ML model performance
    for index in range(1,8): #more than 8 features will probably cause overfitting

        for classifier in clfs:

            print("----------------------------------------------")
            print("----------------------------------------------")
            print(classifier)
            print("----------------------------------------------")
            print("----------------------------------------------")

            sel = RFE(classifier, n_features_to_select = index) # only select most important features to combat overfitting
            sel.fit(X_train, y_train)
            features = X_train.columns[sel.get_support()]
            X_train_rfe = sel.transform(X_train)
            X_test_rfe = sel.transform(X_test)
            print('Selected Feature', index)
            print(features)

            classifier.fit(X_train_rfe, y_train)

            # save the model to disk
            filename = f'./saved_model/{classifier}_{index}_features.sav'
            pickle.dump(classifier, open(filename, 'wb'))

            print(f"{filename} model is saved")

            y_pred= classifier.predict(X_test_rfe)
            scores = cross_val_score(pipeline, X_train_rfe, y_train, cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)) # cross-validation to combat overfitting
            
            print("----------------------------------------------")
            print("TRAIN-TEST")
            print("----------------------------------------------")


            print('confusion matrix', classifier)
            print(confusion_matrix(y_test, y_pred))
            print('classification report')
            print(classification_report(y_test, y_pred))
            print('accuracy score')
            print(accuracy_score(y_test, y_pred))
            

            X_valid_rfe = sel.transform(X_valid)
            y_pred = classifier.predict(X_valid_rfe)

            print("----------------------------------------------")
            print("TRAIN-VALIDATION")
            print("----------------------------------------------")

            print('confusion matrix', classifier)
            print(confusion_matrix(y_valid, y_pred))
            print('classification report')
            print(classification_report(y_valid, y_pred))
            print('accuracy score')
            print(accuracy_score(y_valid, y_pred))




