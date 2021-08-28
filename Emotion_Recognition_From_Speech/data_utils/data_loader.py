import os 
import sys
import pickle
import numpy as np
import librosa; #print("librosa", librosa.__version__)
import tqdm; #print("tqdm", tqdm.__version__)
import soundfile

sys.path.insert(1, '../')

N_EMOTIONS = 8 
PATH = 'dataset/Audio_Speech_Actors_01-24'
ID = ['emotion', 'emotion_intensity', 'statement', 'repetition', 'actor']
EMO_DICT= {0:'neutral', 1:'calm', 2:'happy', 3:'sad', 4:'angry', 5:'fearful', 6:'disgust', 7:'surprised'}

INDEXES = [2,3,4,5,6]


class Data_loader():
    def __init__(self, path=PATH, sampling_rate=16000):
        self.path = path
        self.sampling_rate = sampling_rate

    def _samples_from_pathname(self, pathname):
        """[ Extracts the samples from the .wav audio file in pathname]
        """
        # load with the default sampling freq
        samples, default_sample_rate = librosa.load(pathname, sr=None)
        # resample down to sampling_rate
        samples = librosa.resample(samples, default_sample_rate, self.sampling_rate)

        #samples, default_sample_rate = soundfile.read(pathname)

        return samples

    def _features_from_pathname(self, samples, mfcc, chroma, mel):
        """[ Extracts the samples from the .wav audio file in pathname]
        """
        mfcc_featuers, chr_features, mel_features = None, None, None
        # print(samples.shape)
        if mfcc:
            mfcc_featuers=librosa.feature.mfcc(y=samples, sr=self.sampling_rate, n_mfcc=40).T
        if chroma:
            stft=np.abs(librosa.stft(samples))
            chr_features=librosa.feature.chroma_stft(S=stft, sr=self.sampling_rate).T
        if mel:
            mel_features=librosa.feature.melspectrogram(samples, sr=self.sampling_rate).T

        return mfcc_featuers, chr_features, mel_features

    def _hot_encode(self,i):
        """[One hot encoding for the labels.]

        Arguments:
            i {[int]} -- [ Nr 1-8 which correspond to 01-08 emotion types]

        Returns:
            [lis] -- [One hot encoded vector]
        """
        return np.eye(N_EMOTIONS, dtype=np.int32)[i-1]

    def _ids_from_pathname(self, pathname):
        """[Extracts the IDs from the name of the file in pathname]
        """
        fname= pathname.split('.wav')[0].split('/')[3].split('-')
        dic = {}
        for id, i in zip(ID, INDEXES):
            if id == 'emotion':
                dic[id] = self._hot_encode(int(fname[i]))
                continue
            dic[id]=fname[i]
        return dic


    def _dict_from_pathname(self, pathname, mfcc=True, chroma=False, mel=False):
        """[Creates dict with information about the file in pathname]
        """
        dic = self._ids_from_pathname(pathname)
        samples = self._samples_from_pathname(pathname)
        dic['samples'] = samples
        mfcc_featuers, chr_features, mel_features = self._features_from_pathname(samples, mfcc, chroma, mel)

        if mfcc: dic['mfcc'] = mfcc_featuers
        if chroma: dic['chroma'] = chr_features
        if mel: dic['mel'] = mel_features

        return dic

    def load_data(self, path=None, sampling_rate=None, mfcc=True, chroma=False, mel=False):
        """[The created 'all_data' is a list of dictionaries, one for each speaker]
            
        Arguments:
            pathname {[str]} -- Path to the actors' folders with the recordings

        Returns:
            all_data {[list]} --
            -- List of dictionaries with the following structure: 
                'emotion':              (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)            
                'emotion_intensity':    (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion    
                'statement':            (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door")
                'repetition':           (01 = 1st repetition, 02 = 2nd repetition)
                'actor':                (01 to 24. Odd numbered actors are male, even numbered actors are female).
        """
        if path is not None: self.path = path
        if sampling_rate is not None: self.sampling_rate = sampling_rate

        if os.path.isfile("dataset/all_data.p"): 
            return pickle.load( open( "dataset/all_data.p", "rb" ) )

        print("No prepared dataset is found. We will upload it manually. This will only happen once.\n")
        all_data = []
        for actor in tqdm.tqdm(os.listdir(self.path)):
            path_to_actor = "{}/{}".format(self.path, actor)
            path_to_sounds = [ "{}/{}".format(path_to_actor, f) for f in os.listdir(path_to_actor) if f.endswith('.wav') ]
            all_data += [self._dict_from_pathname(f,mfcc, chroma, mel) for f in path_to_sounds]

        pickle.dump( all_data, open( "dataset/all_data.p", "wb" ) )
        return all_data



## USAGE
def main():
    #dl= Data_loader('dataset/Audio2')
    dl= Data_loader()
    data = dl.load_data(mfcc=True, chroma=True, mel=True)
    print(len(data))

if __name__ == "__main__":
    main()
