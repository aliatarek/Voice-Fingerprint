import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import fft, signal
import scipy
from scipy.io.wavfile import read
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from PyQt5.QtWidgets import QTableWidgetItem

import glob
from typing import List, Dict, Tuple
from tqdm import tqdm
import pickle

from scipy.io import wavfile
import librosa
import numpy as np
import pyaudio
import os
speaker_names=["Amr:Decipher","Amr:Grant","Amr:Permission","Alia:Decipher","Alia:Grant","AliaPermission","Hamza:Decipher","Hamza:Grant","Hamza:Permission","Mahmoud:Decipher","Mahmoud:Grant","Mahmoud:Permission"]
modeflag=1
song_name_index = {} 
similarity_score=[]
y_data=[]
x_data=[]
model=None


def record_audio(duration):
    CHUNK = 1024  # Number of frames per buffer
    FORMAT = pyaudio.paFloat32  # Audio format (32-bit floating-point)
    CHANNELS = 1 # Number of audio channels (1 for mono, 2 for stereo)
    RATE = 44100  # Sample rate
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.float32))

    print("Finished recording")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    return np.hstack(frames), RATE

def create_mfcc(audio_file):
    y, sr = librosa.load(audio_file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chromaprint = librosa.feature.chroma_stft(y=y, sr=sr)
    rms=librosa.feature.rms(y=y)
    mel=librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=256, hop_length=64, n_mels=13))
    rythm=librosa.feature.tempogram(y=y,sr=sr)
    zero_crossing=librosa.feature.zero_crossing_rate(y=y)

    return mfcc, chromaprint,rms,mel,rythm,zero_crossing

def train_model(self):
      global song_name_index
      global similarity_score
      global y_data
      global x_data
      global model
      dataset_directory = "C:/Users/alia/Desktop/Data"
      songs = glob.glob('C:/Users/alia/Desktop/Data/*.wav')
      for index, filename in enumerate(tqdm(sorted(songs))):
        song_name_index[index] = filename[27:38]
        # Read the song, create a constellation and hashes
        Fs, audio_input = read(filename)
        if filename.endswith(".wav"):
                filepath = os.path.join(dataset_directory, filename)
                # mfcc = create_mfcc(filepath)
                # dataset.append(mfcc)
                # file_names.append(filename)
        reference_mfcc_initial,reference_chroma_initial,reference_rms_initial,reference_mel_initial,reference_rythm_initial,reference_zero_initial=create_mfcc(filepath)
        reference_mfcc=preprocesstestData(reference_mfcc_initial,reference_chroma_initial,reference_rms_initial,reference_mel_initial,reference_rythm_initial,reference_zero_initial)
        x_data.append(reference_mfcc)
        y_data.append(song_name_index[index])
        model =RandomForestClassifier(n_estimators=300,criterion="entropy",bootstrap=False,warm_start=True)
        model.fit(x_data, y_data)

def start_recording(self):
     global modeflag
     global y_data
     songs = glob.glob('C:/Users/alia/Desktop/Data/*.wav')
     for index, filename in enumerate(tqdm(sorted(songs))):
        song_name_index[index] = filename[27:35]

     duration = 3  # Recording duration in seconds
     self.recordingStatus.setText("Recording")
     input_audio, fs = record_audio(3)
     self.recordingStatus.setText("Recording Stopped")
     wavfile.write("output_file.wav", fs, ((input_audio/np.max(input_audio)) * 32767).astype(np.int16))  # Scaling to int16 before saving
     user_audio = "output_file.wav"

     train_model(self)

     test_mfcc_initial, test_chroma_initial, test_rms_initial,test_mel_initial,test_rythm_initial,test_zero_initial = create_mfcc(user_audio)
     test_mfcc=preprocesstestData(test_mfcc_initial,test_chroma_initial, test_rms_initial,test_mel_initial,test_rythm_initial,test_zero_initial)
     similarity_scoree = model.predict(test_mfcc.reshape(1,-1))

     class_probabilities = model.predict_proba(test_mfcc.reshape(1,-1))
     if modeflag==1:
             threshold=0.38
             flag=False
             for i in range(5):
                if class_probabilities[0][i]>threshold:
                        flag=True
                        print(class_probabilities[0][i])
             if flag==True:
                        self.AccessStatus.setText("Access Granted")
             else:
                        self.AccessStatus.setText("Access Denied")
             dataa = [str((class_probabilities[0][0])*100)+"%", str((class_probabilities[0][2])*100)+"%", str((class_probabilities[0][1])*100)+"%"]
             for row, data in enumerate(dataa):
                item = QTableWidgetItem(data)
                self.table1.setItem(row-1, 1, item)
             for i in range (len(song_name_index)):
                print(f"{song_name_index[i]}:")
             print(f"Score of {class_probabilities}")
     elif modeflag==2:
              threshold=0.4
              index=0
              flag=None
              for i in range(5):
                if class_probabilities[0][i]>threshold:
                      #class_probabilities[0][i]=threshold
                      flag=True
                      index=i
              value=threshold

              if flag==True:
                # print(self.select_users[index])
                if self.select_users[index]==1:
                 self.AccessStatus.setText(f"Access Granted by {speaker_names[index]}") ##########Adjust with new dataset
                 print (self.select_users[index])
              else:
                self.AccessStatus.setText("Access Denied")
                print (self.select_users[index])
              max_of_three_consecutive = []

              inner_list = class_probabilities[0]  # Accessing the single inner list

              max_numbers = []
              for i in range(0, len(inner_list), 3):  # Loop through indices with a step of 3
               window = inner_list[i:i+3]  # Get a window of three consecutive numbers
               if len(window) == 3:
                        max_numbers.append(max(window))  # Find the maximum within the window
              dataa = [str((max_numbers[1])*100), str((max_numbers[0])*100), str((max_numbers[2])*100),str((max_numbers[3])*100),str(0),str(0),str(0),str(0)]
              for row, data in enumerate(dataa):
                item = QTableWidgetItem(data)
                self.table2.setItem(row-1, 1, item)
              dataa = [str((class_probabilities[0][0])*100)+"%", str((class_probabilities[0][2])*100)+"%", str((class_probabilities[0][1])*100)+"%"]
              for row, data in enumerate(dataa):
                item = QTableWidgetItem(data)
                self.table1.setItem(row-1, 1, item)

              for i in range (len(song_name_index)):
                print(f"{song_name_index[i]}:")
              print(f"Score of {class_probabilities}")
              
def preprocesstestData(test_mfcc,test_chroma,test_rms,test_mel,test_rythm,test_zero):
    test_mfcc_mean=[]
    test_mfcc_var=[]
    test_chroma_mean=[]
    test_chroma_var=[]
    test_rms_mean=[]
    test_rms_var=[]
    test_mel_mean=[]
    test_mel_var=[]
    test_rythm_mean=[]
    test_rythm_var=[]
    test_zero_mean=[]
    test_zero_var=[]
    for i in range(len(test_mfcc)):
            test_mfcc_mean.append(test_mfcc[i].mean())
            test_mfcc_var.append(test_mfcc[i].var())
    for i in range(len(test_chroma)):
            test_chroma_mean.append(test_chroma[i].mean())
            test_chroma_var.append(test_chroma[i].var())
    for i in range(len(test_rms)):
            test_rms_mean.append(test_rms[i].mean())
            test_rms_var.append(test_rms[i].var())
    for i in range(len(test_mel)):
            test_mel_mean.append(test_mel[i].mean())
            test_mel_var.append(test_mel[i].var())
    for i in range(len(test_rythm)):
            test_rythm_mean.append(test_rythm[i].mean())
            test_rythm_var.append(test_rythm[i].var())
    for i in range(len(test_zero)):
            test_zero_mean.append(test_zero[i].mean())
            test_zero_var.append(test_zero[i].var())
    return np.hstack((test_mfcc_mean,test_mfcc_var,test_chroma_mean,test_chroma_var,test_rms_mean,test_rms_var,test_mel_mean,test_mel_var,test_rythm_mean,test_rythm_var,test_zero_mean,test_mel_var))


def compare_mfcc(ref_mfcc,test_mfcc):
    similarity_score=cosine_similarity(ref_mfcc.reshape(1,-1), test_mfcc.reshape(1,-1))
    return similarity_score



# for i in range (len(similarity_score)):
#      print(f"{song_name_index[i]}: Score of {similarity_score[i]}")
        # print(f"{song_name_index[song_id]}: Score of {similarity_score[1]} at {similarity_score[0]}")
    
# for index, filename in enumerate(tqdm(sorted(songs))):
#     song_name_index[index] = filename
#     Y=[]
#     Y.append(song_name_index[index])

# Create the target variable y based on the filenames

# # Loop through each reference sample and print its class probability
# for i, filename in enumerate(song_name_index.values()):
#     # Find the index corresponding to the filename in the training data
#     index_in_training_data = list(song_name_index.values()).index(filename)

#     # Get the class probability for the current reference sample
#     probability_for_sample = class_probabilities[index_in_training_data]

#     print(f"{filename}: Class Probabilities - {probability_for_sample}")
# for i in range (len(song_name_index)):
#    print(f"{song_name_index[i]}:")
# print(f"Score of {class_probabilities}")
# print(f"Score of {class_probabilities[0][1]}")
# print(max(class_probabilities[0]))


      #get index of maximum
      #get checked boxes
        #display probs
        #change second passcode
        #record more and change parameters
      

def whichmode(self,flag):
     global modeflag
     modeflag=flag
     

# similarity_scoree.predict_proba(x_data)
# # similarity_scoree = model.predict_proba([[cosine_similarity([reference_mfcc[0].reshape(1,-1)], [test_mfcc.reshape(1,-1)])]])[0, 1]
# # test_feature = np.array([cosine_similarity(test_mfcc.reshape(1,-1), test_mfcc.reshape(1,-1))])
# # similarity_scoree = model.predict_proba(test_feature)[:, 1]

# #print(f"Similarity Score: {similarity_scoree}")
# for i in range (len(similarity_scoree)):
#    print(f"{song_name_index[i]}: Score of {similarity_score[i]}")
            