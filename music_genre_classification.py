# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 12:01:47 2023

@author: param
"""
import os
import librosa
import numpy as np

def extr(audio_file,genre):

    #print(audio_file)
    # Load the audio file.
    audio, sr = librosa.load(audio_file)

    #chroma
    chroma=librosa.feature.chroma_stft( y=audio, sr=sr, n_fft=2048, hop_length=512,  n_chroma=12)
    chroma_mean=np.mean(chroma)
    chroma_var=np.var(chroma)

    #mfcc10
    mfcc_10 = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=10)
    mfcc_mean10=np.mean(mfcc_10)
    mfcc_var10=np.var(mfcc_10)

    #mfcc11
    mfcc_11 = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=11)
    mfcc_mean11=np.mean(mfcc_11)
    mfcc_var11=np.var(mfcc_11)

    #mfcc12
    mfcc_12 = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=12)
    mfcc_mean12=np.mean(mfcc_12)
    mfcc_var12=np.var(mfcc_12)

    #mfcc13
    mfcc_13 = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean13=np.mean(mfcc_13)
    mfcc_var13=np.var(mfcc_13)


    #mfcc14
    mfcc_14 = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=14)
    mfcc_mean14=np.mean(mfcc_14)
    mfcc_var14=np.var(mfcc_14)


    # Compute the energy of the audio file.
    energy = librosa.feature.rms(y=audio)
    energy_mean=np.mean(energy)
    energy_var=np.var(energy)

    #spectral
    sp_bw=librosa.feature.spectral_bandwidth(y=audio,sr=sr,hop_length=512)
    sp_mean=np.mean(sp_bw)


    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    sp_c_mean=np.mean(spectral_centroid)


    #Spectral_rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    sprf_mean=np.mean(spectral_rolloff)
    sprf_var=np.var(spectral_rolloff)

    # Spectral Flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=audio)
    spflat_mean=np.mean(spectral_flatness)
    spflat_var=np.var(spectral_flatness)




    mean=[mfcc_mean10,mfcc_var10,mfcc_mean11,mfcc_var11,mfcc_mean12,mfcc_var12,mfcc_mean13,mfcc_var13,
          mfcc_mean14,mfcc_var14,energy_mean,energy_var,sp_mean,sp_c_mean,chroma_mean,chroma_var,
          sprf_mean,sprf_var,spflat_mean,spflat_var,genre]
    global mean_values
    mean_values.append(mean)
    #print(mean)



mean_values=[]

# assign directory
directory = "C:/Users/param/OneDrive/Desktop/music/genres_original"


for root, dirs, files in os.walk(directory):
    for name in files:
        print(name)
        filename = os.path.join(root, name)
        genre=root.split('/')[-2]
        
        if genre=='blues':
         extr(filename,1)
         print(filename)
        elif genre=='claassical' :
          extr(filename,2)
        elif genre=='country' :
          extr(filename,3)
        elif genre=='disco' :
          extr(filename,4)
        elif genre=='hiphop' :
          extr(filename,5)
        elif genre=='jazz' :
          extr(filename,6)
        elif genre=='metal' :
          extr(filename,7)
        elif genre=='pop' :
          extr(filename,8)
        elif genre=='reggae' :
         extr(filename,9)   
        elif genre=='rock' :
          extr(filename,10)
        
        




import pandas as pd


DF = pd.DataFrame(mean_values,columns=['mfcc_mean10','mfcc_var10','mfcc_mean11','mfcc_var11','mfcc_mean12','mfcc_var12','mfcc_mean13',
                                       'mfcc_var13','mfcc_mean14','mfcc_var14','energy_mean','energy_var','spectral_bandwidth_mean',
                                       'spectral_centroid mean','chroma_mean','chroma_var','sprf_mean','sprf_var','spflat_mean',
                                       'spflat_var','type'])

# save the dataframe as a csv file
DF.to_csv(r"C:\Users\param\OneDrive\Desktop\music\musicdata.csv")




#building the ML Model
import pandas as pd
from sklearn.model_selection import train_test_split
DF=pd.read_csv('/content/drive/MyDrive/1DTL/footsteps_data.csv')
df=DF[:]


del df['Unnamed: 0']
X=df[['mfcc_mean10','mfcc_var10','mfcc_mean11','mfcc_var11','mfcc_mean12','mfcc_var12',
      'mfcc_mean13','mfcc_var13','mfcc_mean14','mfcc_var14','energy_mean','energy_var',
      'spectral_bandwidth_mean','spectral_centroid mean','chroma_mean','chroma_var',
      'sprf_mean','sprf_var','spflat_mean','spflat_var']]
y=df['type']

X=X.to_numpy()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


print("\nClassification Report:")
print(classification_report(y_test, y_pred))



#TESTING THE REAL TIME AUDIO SIGNAL

import librosa
import numpy as np
import matplotlib.pyplot as plt


def ext(audio_file):


    # Load the audio file.
    audio, sr = librosa.load(audio_file)

    #chroma
    chroma=librosa.feature.chroma_stft( y=audio, sr=sr, n_fft=2048, hop_length=512,  n_chroma=12)
    chroma_mean=np.mean(chroma)
    chroma_var=np.var(chroma)


    #mfcc10
    mfcc_10 = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=10)
    mfcc_mean10=np.mean(mfcc_10)
    mfcc_var10=np.var(mfcc_10)

    #mfcc11
    mfcc_11 = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=11)
    mfcc_mean11=np.mean(mfcc_11)
    mfcc_var11=np.var(mfcc_11)

    #mfcc12
    mfcc_12 = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=12)
    mfcc_mean12=np.mean(mfcc_12)
    mfcc_var12=np.var(mfcc_12)

    #mfcc13
    mfcc_13 = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean13=np.mean(mfcc_13)
    mfcc_var13=np.var(mfcc_13)


    #mfcc14
    mfcc_14 = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=14)
    mfcc_mean14=np.mean(mfcc_14)
    mfcc_var14=np.var(mfcc_14)


    # Compute the energy of the audio file.
    energy = librosa.feature.rms(y=audio)
    energy_mean=np.mean(energy)
    energy_var=np.var(energy)

    #spectral
    sp_bw=librosa.feature.spectral_bandwidth(y=audio,sr=sr,hop_length=512)
    sp_mean=np.mean(sp_bw)


    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    sp_c_mean=np.mean(spectral_centroid)

    #roll_off
    tempo=librosa.feature.tempo(y=audio,sr=sr)
    tempo_mean=np.mean(tempo)
    tempo_var=np.var(tempo)

    #Spectral_rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    sprf_mean=np.mean(spectral_rolloff)
    sprf_var=np.var(spectral_rolloff)

    # Spectral Flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=audio)
    spflat_mean=np.mean(spectral_flatness)
    spflat_var=np.var(spectral_flatness)



    mean=[mfcc_mean10,mfcc_var10,mfcc_mean11,mfcc_var11,mfcc_mean12,mfcc_var12,mfcc_mean13,mfcc_var13,
          mfcc_mean14,mfcc_var14,energy_mean,energy_var,sp_mean,sp_c_mean,chroma_mean,chroma_var,
          sprf_mean,sprf_var,spflat_mean,spflat_var]

    global mean_values
    mean_values.append(mean)
    return mean



mean_values=[]




test='/content/353799__monte32__footsteps_6_dirt_shoe.wav'
a=ext(test)
a=np.array(a)
y=a.reshape(1,22)
predicted=classifier.predict(y)
print(predicted)