# Imports

import librosa #To handle .wav files
from librosa import feature

import numpy as np
import pandas as pd
import os
import time

def extract_features(file):

    """
    Function to extract features of a single .wav file
    : attrib file
    This function will return a dataframe of the extracted features of the selected .wav file
    """

    # check if normal or abnormal
    if 'abnormal' in file:
        operation=1
    else:
        operation=0

    # loading the file, getting y and sr (sample rate)
    y, sr = librosa.load(file)

    # getting S and phase
    S, phase = librosa.magphase(librosa.stft(y=y))

    # features for the DataFrame
    # chroma features have pitch which is probably not usefull
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))

    chroma_cqt = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr))

    chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr))

    melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, S=S))
    melspectrogram_min = np.min(librosa.feature.melspectrogram(y=y, sr=sr, S=S))
    melspectrogram_max = np.max(librosa.feature.melspectrogram(y=y, sr=sr, S=S))
    melspectrogram_sum = librosa.feature.melspectrogram(y=y, sr=sr, S=S).sum()
    melspectrogram_corr= np.mean(np.corrcoef(librosa.feature.melspectrogram(y=y, sr=sr, S=S)))
    melspectrogram_std= np.std(librosa.feature.melspectrogram(y=y, sr=sr, S=S))
    
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr))

    rms = np.mean(librosa.feature.rms(y=y, S=S))

    # spectral centroid computes weighted mean of the frequencies in the sound
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, S=S))

    # bandwidth(blue zone) is the difference between the upper and lower frequencies in a continuous band of frequencies
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, S=S))

    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, S=S))

    # spectral flatness (or tonality coefficient) is a measure to quantify how much noise-like a sound is, as opposed to
    # being tone-like 1. A high spectral flatness (closer to 1.0) indicates the spectrum is similar to white noise. 
    # It is often converted to decibel.

    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y, S=S))

    #The roll-off frequency is defined for each frame as the center frequency for a spectrogram bin such that at 
    #least roll_percent (0.85 by default) of the energy of the spectrum in this frame is contained in this bin and 
    #he bins below. This can be used to, e.g., approximate the maximum (or minimum) frequency by setting roll_percent 
    #to a value close to 1 (or 0). Rolloff with rolloff coefficient 0.01 seems to be the same for (ab)normal
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, S=S))

    # get coefficients of fitting an nth-order polynomial to the columns of a spectrogram
    poly_features = np.mean(librosa.feature.poly_features(y=y, sr=sr, S=S))
    
    tonnetz = np.mean(librosa.feature.tonnetz(y=y, sr=sr))
    
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))

    # split the frequencies into harmonic and percussive waves
    y_harmonic, y_percussive = librosa.effects.hpss(y)


    # putting the extracted features into a dataframe and returning the dataframe
    return pd.DataFrame({ 'melspectrogram' : [melspectrogram],'melspectrogram_min':[melspectrogram_min],
                             'melspectrogram_max':[melspectrogram_max], 'melspectrogram_sum':[melspectrogram_sum],
                             'melspectrogram_corr':[melspectrogram_corr] ,'melspectrogram_std':[melspectrogram_std] ,
                             'mfcc' : [mfcc], 'rms' : [rms],
                            'spectral_centroid' : [spectral_centroid], 'spectral_bandwidth' : [spectral_bandwidth],
                            'spectral_contrast' : [spectral_contrast], 'spectral_flatness ' : [spectral_flatness],
                            'spectral_rolloff' : [spectral_rolloff], 
                            'zero_crossing_rate' : [zero_crossing_rate],"mean harm": np.mean(y_harmonic),
                            "mean perc": [np.mean(y_percussive)],"max harm":[np.max(y_harmonic)],"max perc": [np.max(y_percussive)],
                             "min harm":[np.min(y_harmonic)], "min perc":[np.min(y_percussive)], 'normal(0)/abnormal(1)':[operation]})


def create_csv(machine, dB_level):

    """
    Function to extract features of a single .wav file
    : attrib file
    This function will return a dataframe of the extracted features of the selected .wav file
    """
    
    # start measuring excecution time
    start_time = time.time()
    #create empty list that will be filled with pathnames
    #find paths normal wav files
    list_normal_6 = []
    num=[0,2,4,6]
    for i in num:
        directory_normal_6 = f"/home/regis/Desktop/Sound Project/files/{machine}/{dB_level}_dB_{machine}/{machine}/id_0{i}/normal/"
        for filename in os.listdir(directory_normal_6):
            file = f"{directory_normal_6}{filename}"
            list_normal_6.append(file)
    list_normal_6.sort()
    #find paths abnormal wav files
    list_abnormal_6 = []
    num=[0,2,4,6]
    for i in num:
        directory_abnormal_6 = f"/home/regis/Desktop/Sound Project/files/{machine}/{dB_level}_dB_{machine}/{machine}/id_0{i}/abnormal/"
        for filename in os.listdir(directory_abnormal_6):
            file = f"{directory_abnormal_6}{filename}"
            list_abnormal_6.append(file)
    list_abnormal_6.sort()
    #add normal to df
    for wav_file in list_normal_6:
        df = extract_features(wav_file)
        if wav_file == list_normal_6[0]:
            df.to_csv(f'Librosa_features_{machine}_{dB_level}.csv')
        else:
            df.to_csv(f'Librosa_features_{machine}_{dB_level}.csv', mode='a', header=False)
    #add abnormal to df
    for wav_file in list_abnormal_6:
        df = extract_features(wav_file)
        df.to_csv(f'Librosa_features_{machine}_{dB_level}.csv', mode='a', header=False) 
    print("--- %s seconds ---" % (time.time() - start_time))