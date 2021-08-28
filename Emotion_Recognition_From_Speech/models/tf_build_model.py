
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import  LearningRateScheduler

from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os


EMO_DICT= {0:'neutral', 1:'calm', 2:'happy', 3:'sad', 4:'angry', 5:'fearful', 6:'disgust', 7:'surprised'}

class model_builder():
    def __init__(self,data=None,N_splt=None, name=None):

        self.N_splits = N_splt
        self.Name = name
        self.data = data
        
    """
    Our Networks models: 
    """

    def dataFlair(self,arg,c):
        self.Name= "dataflair"

        arg.load_params(self.Name,c)
       
        Model = tf.keras.Sequential([layers.Dense(arg.Units,input_shape=[arg.inputShape], kernel_initializer="he_normal", activation="relu") 
                             ,layers.Dense(arg.target_class_no,activation='softmax')])

        optimizer = tf.keras.optimizers.Adam(learning_rate=arg.learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=arg.Decay)
        Model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
                
        Model.summary()
        return Model
    
    
    def CNN1D(self,arg,c):
        self.Name= "CNN1D"
        arg.load_params(self.Name,c)


        a = int(arg.filters/2) # half of filters
        b = int(arg.filters/4)  # fourth of filters

        model = tf.keras.Sequential()
        model.add(layers.Conv1D(arg.filters, arg.kernel_size, padding='same',input_shape=(arg.inputShape,1), name="C1"))  
        model.add(layers.Activation('relu'))

        model.add(layers.Conv1D(arg.filters, arg.kernel_size, padding='same', name="C2"))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(arg.dropout))
        model.add(layers.MaxPooling1D(pool_size=(arg.pool_size)))

        for i in range(3):
            model.add(layers.Conv1D(a, arg.kernel_size, padding='same', name="C"+str(i+3)))
            model.add(layers.Activation('relu'))

        model.add(layers.Conv1D(a, arg.kernel_size, padding='same', name="C6"))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(arg.dropout))
        model.add(layers.MaxPooling1D(pool_size=(arg.pool_size)))

        model.add(layers.Conv1D(b, arg.kernel_size, padding='same', name="C7"))
        model.add(layers.Activation('relu'))

        model.add(layers.Conv1D(b, arg.kernel_size, padding='same', name="C8"))
        model.add(layers.Activation('relu'))
        model.add(layers.Flatten())

        model.add(layers.Dense(arg.target_class_no, name="OUT")) # Target class number
        model.add(layers.Activation('softmax'))
        model.summary()
        
        # compile model
        opt = tf.keras.optimizers.RMSprop(learning_rate=arg.learningRate, decay=arg.Decay)
        model.compile(optimizer=opt,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        
        return model

    def CNN2D(self):
        raise KeyError("Not yet implemented")
        return 0 
    

    
    def LSTM(self, arg,c):
        self.Name= "LSTM"
        arg.load_params(self.Name,c)
        
        model = tf.keras.Sequential()
        
        model.add( tf.keras.layers.LSTM(arg.lstm_size, input_shape=(None, arg.input_size)))

        # Add a Dense layer with 64 units.
        model.add(layers.Dense(units=arg.Units, kernel_initializer="he_normal"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(arg.target_class_no, activation='softmax'))
        model.summary()

        opt = tf.keras.optimizers.Adam(learning_rate=arg.learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=arg.Decay, amsgrad=False, clipnorm=1.)
       
        model.compile(loss='categorical_crossentropy', 
                        optimizer=opt,
                        metrics=['accuracy'])
        
        return model

 

    """
    Cross validation performed: 
    """


    def confusion_matrix(self,targets, predictions,arg, fold_no,labels,acc):

        
        ground_truth = np.argmax(targets,axis=1)
        pred = np.argmax(predictions,axis=1)
        label_idx = np.arange(0,len(labels))
        confusionMatrix = confusion_matrix(ground_truth, pred, labels=label_idx)
        
        path = "results/CM/"+str(self.Name)+ str(fold_no) +"confusionMat.p"
        path2fig = "results/CM/"+str(self.Name)+ str(fold_no) + "confusionMat_fig.eps"
        path2 = "results/CM/"+str(self.Name)+ str(fold_no) +"confusionMat_labels.p"
        pickle.dump( confusionMatrix, open( path, "wb" ) )
        pickle.dump(labels, open(path2,"wb"))
        
        fig = plt.figure()   
        sns.heatmap(confusionMatrix,annot=True, fmt=".1f")
        plt.title("Confusion Matrix of the classifier \n Model: "+str(self.Name)+ " with an accuracy of " + str(np.round(acc,2)))
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels,rotation=-45)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.ioff()
        plt.show()
        fig.savefig(path2fig)     
        
        return 0

    def save_data(self, model, CM):
        # 1) plot kfold cross
        # 2) confusion matrix plot
            
        # pickle.dump()
        return 0


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

    def transform_data(self,mode, X,Y):
        if mode ==0: 
            X, Y_train = np.array(X), np.array(Y)
            X = tf.keras.preprocessing.sequence.pad_sequences(X)
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            X_train = np.nan_to_num((X - mean)/std, nan=0.0)
            
        elif mode == 1: 
            X = np.expand_dims(np.array(X), axis=2)
            Y_train = np.array(Y)
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            X_train = (X - mean)/std

        elif mode == 2:
            X_train, Y_train = np.array(X), np.array(Y)
        else: 
            raise KeyError("Decide how to transform data")

        return X_train,Y_train


    def run(self, model, emotions, features, arg):
        
        seed = 7
        np.random.seed(seed)
        mode = None
        if self.Name == None: raise KeyError("Name your network.")


        elif self.Name == "LSTM":
            c = None
            arg.load_params(self.Name,c)
            VF = False
            splitter = TimeSeriesSplit(n_splits=arg.N_splits)
            mode = 0
            
        elif self.Name == "dataflair":
            c = None
            arg.load_params(self.Name,c)
            VF = True
            splitter = KFold(n_splits=arg.N_splits, shuffle=True, random_state=seed)
            mode = 2
        elif self.Name == "CNN1D":
            c = None
            arg.load_params(self.Name,c)
            VF = True
            splitter = KFold(n_splits=arg.N_splits, shuffle=True, random_state=seed)
            mode = 1
       
        else:
            raise KeyError('Name your Network [2].')

        
        X, Y = self.create_dataset(emotions, features, vector_feature=VF)
        inputs, targets =self.transform_data(mode, X,Y)
    
        
        CV_scores = []
        acc_per_fold = []
        loss_per_fold = []
        fold_no =1
        acc =0
        fig = plt.figure()
        for train, test in splitter.split(X,Y):
            
            Model = model
            
            # Generate a print
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ')
            model_history=Model.fit(inputs[train], targets[train], batch_size=16, epochs=arg.EPOCHS)

            plt.subplot(121)
            
            plt.plot(model_history.history['loss'],label="fold:" +str(fold_no))
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.subplot(122)
            plt.plot(model_history.history['accuracy'],label="fold:" +str(fold_no))
            plt.ylabel('Acc')
            plt.xlabel('Epochs')
           
            #evaluate model
            prediction = Model.predict(inputs[test])
           
            scores = Model.evaluate(inputs[test], targets[test], verbose=0)
            print(f'Score for fold {fold_no}: {Model.metrics_names[0]} of {scores[0]}; {Model.metrics_names[1]} of {scores[1]*100}%')
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])
            fold_no+=1

        av_loss = np.mean(loss_per_fold)
        av_acc = np.mean(acc_per_fold) 
        plt.suptitle(f'Cross validation, model: {str(self.Name)}\n Average (loss={str(np.round(av_loss,2))}, acc={str(np.round(av_acc,2))}).\n  ')
        plt.legend()
        
        path2fig = "results/CV/"+str(self.Name)+ str(fold_no) + "crossVal_fig.eps"
        fig.savefig(path2fig)   

        self.confusion_matrix(targets[test],prediction,arg,int(fold_no-1), labels=emotions, acc= av_acc)
        print("Kfold average:\n loss:", av_loss, "\n accuracy:", av_acc)
            

def create_features():
        allfeatures = ['mfcc', 'chroma', 'mel']
        all_combinations = []
        for i in range(len(allfeatures)+1):
            combinations_object = itertools.combinations(allfeatures, i)
            combinations_list = list(combinations_object)
            all_combinations += combinations_list
        return all_combinations[1:]

        

def create_emotions():
    
    observed_emotions = [str(EMO_DICT[idx]) for idx in range(len(EMO_DICT))]

    observed_emotions2 = ['calm', 'happy', 'fearful', 'disgust']
    
    """
    if we want all combinations:

    all_combinations = []
    for i in range(len(observed_emotions)+1):
        combinations_object = itertools.combinations(observed_emotions, i)
        combinations_list = list(combinations_object)
        all_combinations += combinations_list
    
    """

    return [observed_emotions,observed_emotions2]


class ARG():
    def __init__(self, test_size):
        self.test_size = 0.2

        self.learningRate = None
        self.target_class_no = None
        self.Units = None
        self.inputShape = None
        self.Decay = None
        self.batch_size = None
        self.pool_size = None
        self.dropout = None
        self.filters = None
        self.kernel_size = None
        self.lstm_size = None

        self.N_splits = None
        self.EPOCHS = 10
        

    def dataflair_params(self, emSz):
        self.learningRate=0.003 
        self.target_class_no = emSz
        self.Units=300 
        self.inputShape = 180
        self.Decay = 0.1
        self.N_splits = 10
        self.EPOCHS = 50
        
    
    def CNN1D_params(self, emSz):
        self.target_class_no = emSz
        self.filters = 256
        self.pool_size = 4
        self.kernel_size = 8
        self.learningRate = 0.00001
        self.inputShape = 180
        self.dropout = 0.25
        self.Decay = 1e-6
        self.N_splits = 10
        self.EPOCHS = 5
        
    
    def LSTM_params(self, emSz):
        self.target_class_no = emSz
        self.lstm_size = 300
        self.Units = 64
        self.input_size = 180
        self.learningRate = 0.0001
        self.Decay=1e-6
        self.N_splits = 10
        self.EPOCHS = 5
        

    def load_params(self, name, emSz):
        if name == "LSTM": 
            self.LSTM_params(emSz)

        elif name == "dataflair":
            self.dataflair_params(emSz)

        elif name == "CNN1D":
            self.CNN1D_params(emSz)
        else: 
            raise KeyError("Name not found.")
        return 0 





    
def main():

    # To test the network in this file and output the model summary:

    m = model_builder()
    emotions_list = create_emotions()
    features_list = create_features()
    arg = ARG(test_size= 0.2)
    c = len(emotions_list[1])
    
    
    print('dataflair\n')
    m.dataFlair(arg,c)
    print('CNN1D\n')
    m.CNN1D(arg,c)
    print('LSTM\n')
    m.LSTM(arg,c)
    

    """
    Run this in the main file to test if it works for cross validaiton: 
    

    m = model_builder(data)
    emotions_list = create_emotions()
    features_list = create_features()
    arg = ARG(test_size= 0.2)
    c = len(emotions_list[1])
   
    
    m.run(m.LSTM(arg,c),emotions_list[1],['mfcc', 'chroma', 'mel'],arg)
    #m.run(m.CNN1D(arg,c),emotions_list[1],['mfcc', 'chroma', 'mel'],arg)
    #m.run(m.dataFlair(arg,c),emotions_list[1],['mfcc', 'chroma', 'mel'],arg)
    """
    
    
    
    
    

    

if __name__ == "__main__":
    main()


