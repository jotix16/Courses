import matplotlib.pyplot as plt
import librosa
import librosa.display as disp
import random
import pandas as pd
import numpy as np

from data_utils.data_loader import Data_loader, EMO_DICT
from data_utils.tf_dataset_creator import get_dataset
from models.tf_build_model import model_builder, ARG, create_emotions, create_features
from sklearn.model_selection import train_test_split

def main():
    # load data
    dl = Data_loader()
    data = dl.load_data()

    # trainSet, testSet = train_test_split(data,test_size=0.2,random_state=1337 )
    # print("\nSPITT: ", "train:", len(trainSet)/len(data), "test:", len(testSet)/len(data), "\n")

    # # Create panda dataframe for the whole dataset
    # df = pd.DataFrame(data)
    # print("--Full dataset statistics:")
    # print(df.drop(columns=['samples', 'mfcc', 'chroma', 'mel']).astype(str).describe(), "\n")

    # # Dataframes for train valid and test
    # print("--Training set statistics:")
    # print(pd.DataFrame(trainSet).drop(columns=['samples', 'mfcc','chroma', 'mel']).astype(str).describe(), "\n")

    # print("--Test set statistics:")
    # print(pd.DataFrame(testSet).drop(columns=['samples', 'mfcc','chroma', 'mel']).astype(str).describe())


    ## PLOT RAW AUDIO
    # for i in range(30):
    #     print(i,  EMO_DICT[np.argmax(data[i]['emotion'])]   )

    plot_raw_audio = False
    if plot_raw_audio:
        emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        indexes = [13, 5, 23, 16, 14, 6, 17, 7]
        sr = 16000
        plt.figure(figsize=(10,6))
        for i, ind in enumerate(indexes):     
            plt.subplot(4,2,i+1)
            disp.waveplot(data[ind]['samples'], sr, alpha=0.8, x_axis='s')
            
            locs, _ = plt.xticks()
            tixlabels = [ str(tic)+'s' for tic in locs]
            plt.xticks(locs, tixlabels)

            plt.xlabel("")
            plt.ylabel("Amplitude")
            plt.title( EMO_DICT[np.argmax(data[ind]['emotion'])])

        plt.tight_layout()
        plt.show()

    # Plot MEL for the emotion happy
    plot_mel_features = True
    if plot_mel_features:
        plt.figure(figsize=(9, 3))
        disp.specshow(librosa.power_to_db(data[23]['mel'], ref=np.max).T, x_axis='time')
        plt.colorbar()
        plt.show()


    visualize = False
    if visualize:
        plt.figure(figsize=(10, 4))
        k=0
        for i in range(3):
            k = random.randint(0, len(data))

            plt.subplot(3,3,3*i+1)
            disp.specshow(data[k]['mfcc'].T, x_axis='time')
            plt.colorbar()
            plt.title('MFCC '+ EMO_DICT[np.argmax(data[k]['emotion'])])

            plt.subplot(3,3,3*i+2)
            disp.specshow(librosa.power_to_db(data[k]['mel'].T, ref=np.max), x_axis='time')
            plt.colorbar()
            plt.title('MEL '+ EMO_DICT[np.argmax(data[k]['emotion'])])

            plt.subplot(3,3,3*i+3)
            disp.specshow(data[k]['chroma'].T, x_axis='time')
            plt.colorbar()
            plt.title('CHROMA '+ EMO_DICT[np.argmax(data[k]['emotion'])])
        plt.tight_layout()
        plt.show()    

if __name__ == "__main__":
    main()