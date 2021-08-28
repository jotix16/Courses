#%%

import tensorflow as tf
from tensorflow.keras.callbacks import  LearningRateScheduler
from models import Models

from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, confusion_matrix

import sys
sys.path.append(".")
from data_utils import data_loader
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


EMO_DICT= {0:'neutral', 1:'calm', 2:'happy', 3:'sad', 4:'angry', 5:'fearful', 6:'disgust', 7:'surprised'}

class simulator():
    def __init__(self, data=None):
        self.data = data
        
    def create_dataset(self, observed_emotions, features_list, vector_feature = False):

        N_observed = len(observed_emotions)
        NR_TO_NR= {}
        for i,obs in enumerate(observed_emotions):
            for idx, emotion in EMO_DICT.items():
                if obs == emotion:
                    NR_TO_NR[idx] = i
        x = []
        y = []
      
        for d in self.data:
            emot_nr = np.argmax(d['emotion'])
            if EMO_DICT[emot_nr] in observed_emotions:
                feat = d[features_list[0]]

                for i in range(1,len(features_list)):
                    feature = features_list[i]
                    feat = np.hstack((feat,d[feature]))
                    
                if vector_feature:
                    feat = np.mean(feat, axis=0)
                x.append(feat)
                y.append(np.eye(N_observed, dtype=np.int32)[NR_TO_NR[emot_nr]-1])
        return x,y

    def run_all(self, model_names, observed_emotions, features_list):
        for name,features in zip(model_names,features_list):
            self.pars_and_run(name, observed_emotions, features)

    def pars_and_run(self, model_name, observed_emotions, features_list):
        # Create the training data
        ## For dataflair we take the man
        vector_feature = False
        if model_name in ['MLP', 'CNN1D']: vector_feature = True
        X, Y = self.create_dataset(observed_emotions, features_list, vector_feature=vector_feature)
        X, Y =  np.array(X), np.array(Y)


        N = len(observed_emotions)

        seed = 7
        n_splits = 10
        EPOCHS = 150 
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        if model_name == 'MLP': 
            input_shape= X.shape[1]
            # print(X.shape)
            # return
            model = Models.DataFlair(input_shape=input_shape, target_class_no=N)

        elif model_name == 'CNN1D':
            input_shape= X.shape[1]
            X = np.expand_dims(np.array(X), axis=2)
            mean = np.mean(X, axis=0); std = np.std(X, axis=0)
            X = (X - mean)/std
            # input_shape= X.shape[2] 
            model = Models.CNN1D(input_shape=input_shape, target_class_no=N)

        elif model_name == 'CNN1DAUDIO': ##FROM PAPER
            ## CAN normalize too
            # X are audio samples 
            X = np.expand_dims( tf.keras.preprocessing.sequence.pad_sequences(X) ,axis = 2)
            mean = np.mean(X, axis=0); std = np.std(X, axis=0)
            X = np.nan_to_num((X - mean)/std, nan=0.0)
            
            input_shape = (X.shape[0],1)
            model = Models.CNN1D_AUDIO(input_shape=input_shape, target_class_no=N )

        elif model_name == 'CNN2D': ##FROM PAPER(normally used with mel)
            ## CAN normalize too 
            X = np.expand_dims( tf.keras.preprocessing.sequence.pad_sequences(X) ,axis = 3)
            input_shape = (X.shape[1],X.shape[2],1)
            model = Models.CNN2D(input_shape=input_shape, target_class_no=N )

        elif model_name == 'LSTM':
            # pad
            model = Models.LSTM(lstm_size=300, Units=64,  input_shape= 180, target_class_no=N)
            X = tf.keras.preprocessing.sequence.pad_sequences(X)
            # Normalize
            mean = np.mean(X, axis=0); std = np.std(X, axis=0)
            X = np.nan_to_num((X - mean)/std, nan=0.0)

            splitter = TimeSeriesSplit(n_splits=n_splits)
        else: raise KeyError

        name_of_model = model_name+"_emot"+str(N)+"_feat"+str(len(features_list))
        self.kfold( model, name_of_model , X,Y, EPOCHS, splitter, observed_emotions)

    def kfold(self, model, model_name, X, Y, EPOCHS, splitter, observed_emotions):
        CV_scores = []
        acc_per_fold_test = []
        loss_per_fold_test = []
        acc_per_fold_val = []
        loss_per_fold_val = []
        fold_no =1

        plt.figure(figsize=(10,6))
        sns.set()
        savepath = "backup/"+model_name+"_model"
        model_name = model_name.split("_")[0]

        model.save(savepath) # saves compiled state

        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=1337)
        Callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        max_test = 0
        max_test_model = tf.keras.models.load_model(savepath)
        for idx_train, idx_val in splitter.split(X_train,Y_train):
            
            Model = tf.keras.models.load_model(savepath)
            
            # Generate a print
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ')
            model_history=Model.fit(
                X_train[idx_train],
                Y_train[idx_train],
                batch_size=16,
                epochs=EPOCHS,
                shuffle=True,
                validation_data=(X_train[idx_val], Y_train[idx_val]),
                callbacks=[Callback])

            plt.subplot(121)
            plt.plot(model_history.history['loss'],label="training data, fold:" +str(fold_no))
            plt.plot(model_history.history['val_loss'],':',label="validation data, fold:" +str(fold_no))
            plt.title('Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')

            
            plt.subplot(122)
            plt.plot(model_history.history['accuracy'],label="training data, fold:" +str(fold_no))
            plt.plot(model_history.history['val_accuracy'],':',label="validation data, fold:" +str(fold_no))
            plt.title('Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            

            #evaluate model
            scores_test = Model.evaluate(X_test,  Y_test, verbose=0)
            scores_val = Model.evaluate(X_train[idx_val],  Y_train[idx_val], verbose=0)
            print(f'Validation Score for fold {fold_no}: {Model.metrics_names[0]} of {scores_val[0]}; {Model.metrics_names[1]} of {scores_val[1]*100}%')
            print(f'Test Score for fold {fold_no}: {Model.metrics_names[0]} of {scores_test[0]}; {Model.metrics_names[1]} of {scores_test[1]*100}%')
            

            # update max test model
            if scores_val[1]>max_test:
                max_test = scores_val[1]
                max_test_model.set_weights(Model.get_weights())


            acc_per_fold_val.append(scores_val[1] * 100)
            loss_per_fold_val.append(scores_val[0])

            acc_per_fold_test.append(scores_test[1] * 100)
            loss_per_fold_test.append(scores_test[0])
            
            fold_no+=1

        plt.suptitle("Model: " +model_name)
        av_loss_test = np.mean(loss_per_fold_test)
        av_acc_test = np.mean(acc_per_fold_test) 
        av_loss_val = np.mean(loss_per_fold_val)
        av_acc_val = np.mean(acc_per_fold_val) 

        
        pickle.dump( av_loss_test, open( savepath+"/average_loss_test.p", "wb" ) )
        pickle.dump( av_acc_test, open(savepath + "/average_acc_test.p", "wb" ) )
        pickle.dump( av_loss_val, open( savepath +"/average_loss_val.p", "wb" ) )
        pickle.dump( av_acc_val, open( savepath +"/average_acc_val.p", "wb" ) )


        #plt.legend()
        plt.savefig(savepath+'/crossVal_fig.png')
        plt.savefig(savepath+'/crossVal_fig.eps')
        #plt.show()
        
        x_axis = np.arange(fold_no-1)
        ## Accuracy and validation for each fold
        plt.figure()
        plt.suptitle("Model: " +model_name)
        plt.subplot(121)
        plt.scatter(x_axis,acc_per_fold_test, label="test set")
        plt.scatter(x_axis,acc_per_fold_val, label="validation set")
        plt.title('Final score for each fold')
        plt.xlabel('k-fold')
        
        plt.subplot(122) 
        plt.scatter(x_axis, loss_per_fold_test, label="test set")
        plt.scatter(x_axis, loss_per_fold_val, label="validation set")
        plt.title('Final loss for each fold')
        plt.xlabel('k-fold')
        plt.legend()
        
        plt.savefig(savepath + "/final_score_and_loss_kfold.eps")
        plt.savefig(savepath + "/final_score_and_loss_kfold.png")


        ## Confusion matrixc
        prediction = max_test_model.predict(X_test)
        self.confusion_matrix(Y_test,prediction, int(fold_no-1), labels=observed_emotions, acc= av_acc_val, savepath=savepath, modelname=model_name)
        print("Validation data, Kfold average:\n loss:", av_loss_val, "\n accuracy:", av_acc_val)
        print("Test data, Kfold average:\n loss:", av_loss_test, "\n accuracy:", av_acc_test)

    def confusion_matrix(self, targets, predictions, fold_no, labels, acc, savepath, modelname):
        ground_truth = np.argmax(targets,axis=1)
        pred = np.argmax(predictions,axis=1)
        label_idx = np.arange(0,len(labels))
        confusionMatrix = confusion_matrix(ground_truth, pred, labels=label_idx)
        
        
        path2figeps = savepath + "/confusionMat_fig.eps"
        path2figpng = savepath + "/confusionMat_fig.png"
        
        fig = plt.figure()   
        sns.heatmap(confusionMatrix,annot=True, fmt=".1f")
        plt.title("Confusion Matrix of the classifier \n Model: "+ modelname)
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels,rotation=-45)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.ioff()
        plt.show()
        fig.savefig(path2figeps)     
        fig.savefig(path2figpng)  
        return 0

def main():
    # %%
    # load data
    dl= data_loader.Data_loader()
    data = dl.load_data(mfcc=True, chroma=True, mel=True)

    # the mode
    m = simulator(data)
    # %%
    # CNN2D
    observed_emotions2 = ['calm', 'happy', 'fearful', 'disgust']
    #m.run_all(['CNN2D'], observed_emotions2, [['mel']])
    #m.run_all(['CNN1DAUDIO'], observed_emotions2, [['samples']])

    # ## LSTM
    # observed_emotions = [str(EMO_DICT[idx]) for idx in range(len(EMO_DICT))]
    m.run_all(['MLP'], observed_emotions2, [['mfcc','chroma','mel']])



def test_1():
    # load data
    dl= data_loader.Data_loader()
    data = dl.load_data(mfcc=True, chroma=True, mel=True)

    # the mode
    m = simulator(data)

    observed_emotions = [str(EMO_DICT[idx]) for idx in range(len(EMO_DICT))]

    print('\nALL EMOTIONS & ALL FEATURES')
    print(observed_emotions)
    print('\nTESTING MLP')
    m.run_all(['MLP'], observed_emotions, [['mfcc','chroma','mel']])

    print('\nALL EMOTIONS & ALL FEATURES')
    print('TESTING CNN1D')
    m.run_all(['CNN1D'], observed_emotions, [['mfcc','chroma','mel']])

    print('\nALL EMOTIONS & ALL FEATURES')
    print('TESTING LSTM')
    m.run_all(['LSTM'], observed_emotions, [['mfcc','chroma','mel']])



    observed_emotions2 = ['calm', 'happy', 'fearful', 'disgust']
    print('\nSUBSET OF EMOTIONS & ALL FEATURES')
    print(observed_emotions2)
    print('\nTESTING MLP')
    m.run_all(['MLP'], observed_emotions2, [['mfcc','chroma','mel']])

    print('\nALL EMOTIONS & ALL FEATURES')
    print('TESTING CNN1D')
    m.run_all(['CNN1D'], observed_emotions2, [['mfcc','chroma','mel']])

    print('\nALL EMOTIONS & ALL FEATURES')
    print('TESTING LSTM')
    m.run_all(['LSTM'], observed_emotions2, [['mfcc','chroma','mel']])





if __name__ == "__main__":
    main()


