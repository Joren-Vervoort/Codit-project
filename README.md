# Machine conditions monitoring

- Type of challenge: Learning
- Duration: `2 weeks`
- Deadline : `02/04/2021 09:00 AM`
- Team challenge: Group project

## Mission objectives

- Be able to work and process data from audio format
- Find insights from data, build hypothesis and define conclusions
- Build machine learning models for predictive classification and/or regression
- Select the right performance metrics for your model
- Evaluate the distribution of datapoints and evaluate its influence in the model
- Be able to identify underfitting or overfitting that might exist on the model
- Tuning parameters of the model for better performance
- Select the model with better performance and following your
  customer's requirements
- Define the strengths and limitations of the model


## The Mission

Acme Corporation is a worldwide supplier of technological equipment. The factory is facing significant problems with their manufacturing line, the machines are constantly facing failures due to a lack of maintenance and the production is stopped every time an unexpected failure is presented. As a result, Acme is losing millions of U.S Dollars and important clients like Wile E. Coyote are experiencing delays in deliveries. 

The company has collected audio samples of equipment working on normal and anomalous conditions. Their objective is to develop a machine learning model able to monitor the operations and identify anomalies in the sound pattern.

The implementation of this model can allow Acme to operate the manufacturing equipment at full capacity and detect signs of failure before the damage is so critical that the production line has to be stopped.

Your mission is to build a machine learning model for Acme so they can continue their manufacturing activities.

### Must-have features

- Explanatory graphics of insights found in data
- Implementation of machine learning models according with the client's requirements
- The performance metrics of the model must be clearly defined.
- Evaluation of the model's performance and definition of its limitations


## Description

This repository includes the following files:
- README.md
- Preprocessing.ipynb
- ML feature & model selection.ipynb
- ML predictor
- Librosa_features (map)
    - Librosa_features_fan_-6dB.csv
    - Librosa_features_fan_0dB.csv
    - Librosa_features_fan_6dB.csv
    - Librosa_features_pump_-6dB.csv
    - Librosa_features_pump_0dB.csv
    - Librosa_features_pump_6dB.csv
    - Librosa_features_slider_-6dB.csv
    - Librosa_features_slider_0dB.csv
    - Librosa_features_slider_6dB.csv
    - Librosa_features_valve_-6dB.csv
    - Librosa_features_valve_0dB.csv
    - Librosa_features_valve_6dB.csv

## Installation

- virtual environment: python 3.7
- pandas                    1.2.3
- numpy                     1.19.2
- matplotlib                3.3.4
- scikit-learn              0.24.1
- imbalanced-learn          0.8.0
- librosa                   0.8.0
- pickleshare               0.7.5


## How to use

The machine learning model is trained and saved in the file XXXXXX. If the user loads the .wav file and specifies the type of machine, the ML model wil predict if a machine is either running normal or abnormal.

## Implementation

The next step could be creating an API using for example Flask. With this API in place, a live datastream could be fed to the machine learning model to live predict the status of the running machines. A warning could be given to an engineer if a machine runs abnormal for a certain time. 

To monitor the machine learning model an interactive API could confirm if a machine that is labeled as 'running abnormal', was really malfunctioning or not. Also if the model missed a malfunctioning machine, the engineer could let the model know. This way the model can be evaluated and maybe fine-tuned later on or dismissed if preforming badly.

## The process

### The dataset

To train, test and validate the model you can download the dataset following this link: https://zenodo.org/record/3384388#.YGZIjk7iuUk

The dataset contains sound recordings for 4 types of machines. For each machine, 3 levels of white noise were added (None, +6dB and -6dB) to simulate environment sounds. For each machine, sound recording of 4 independent units are available. The sound recording are compressed in .zip files.

This dataset is made available by Hitachi, Ltd. under a Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license.This version "public 1.0" contains four models (model ID 00, 02, 04, and 06). 


### Preprocessing

In the Preprocessing.ipynb file the different features of the .wav files are extracted and stored in a dataframe for each kind of added noise (None, +6dB and -6dB), per machine. An extra column 'normal(0)/abnormal(1)' is also added to confirm if the datarow is a normal (given value 0) or abnormal (given value 1). The dataframe is automatically saved as a .csv file. A name is assigned to each created .csv file with the name of the machine and the added noise (f.e.: Librosa_features_pump_-6dB.csv). If you run this for all machines of the dataset, you will get the following 12 .csv files:

- Librosa_features_fan_-6dB.csv
- Librosa_features_fan_0dB.csv
- Librosa_features_fan_6dB.csv
- Librosa_features_pump_-6dB.csv
- Librosa_features_pump_0dB.csv
- Librosa_features_pump_6dB.csv
- Librosa_features_slider_-6dB.csv
- Librosa_features_slider_0dB.csv
- Librosa_features_slider_6dB.csv
- Librosa_features_valve_-6dB.csv
- Librosa_features_valve_0dB.csv
- Librosa_features_valve_6dB.csv


### Machine Learning model selection

The final steps of making the data 'machine-learn-ready' is to concatenate the different noise levels per machine and to balance the dataframe. Because of the imbalance in the dataset, undersampling is applied to balance the data. After this the data is split into a train, test and validation set. 

First a pipeline is created. The different classifiers that will be tested are added and fitted one by one. For each of the classifiers RFE (recursive feature elimination) is  applied to get the most important feateres for each model and a cross_validation is applied to combat overfitting. After checking the metrics using a classification report (precision, recall and f1-score), the best classifiers are hand-picked and selected. This model is then trained and saved using the pickle package.

Now the model is ready to use to predict if a machine is running normal or abnormal using a .wav file.

### Result evalution

With a high precision, recall and f1-score, the model seems to perform good. Altough the data should be tested on predicting extra .wav files to be sure. Also extra noise levels could be added to recreate a more realistic scenario. The best way to be sure is ofcourse to work with a more balanced, more .wav files and more units of each machine.

THANK YOU FOR READING!

