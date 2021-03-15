# -*- coding: utf-8 -*-
import zipfile
import json
import pickle, gzip
import os
import re
import numpy as np
import random
from tqdm import tqdm
from RNN import RNN

# PATHS
trump_dir_path_default = os.path.split(os.path.realpath(__file__))[0]
trump_raw_path_default = os.path.join(trump_dir_path_default, "raw_trump")
trump_ex_path_default = os.path.join(trump_dir_path_default, "ex_trump")

def extract_zip_all(raw_path=trump_raw_path_default,
                    ex_path=trump_ex_path_default):
    
    for root, dirs, files in os.walk(raw_path):
        for file in files:
            tmp_path = os.path.join(root, file)
            if not zipfile.is_zipfile(tmp_path):
                continue
            if "condensed" in tmp_path:
                continue
            zf = zipfile.ZipFile(tmp_path, mode="r")
            zf.extractall(ex_path)
    
def read_json_text_all(dir_path=trump_ex_path_default):
    
    data = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            print(file)
            tmp_path = os.path.join(root, file)
            with open(tmp_path, "r") as f:
                tmp_data = json.load(f)
            for d in tmp_data:
                if "text" in d.keys():
                    data.append(d["text"]+"<")

    return data
    
def get_letter_dict(data, threshold=89):
    
    cnt = {}
    for s in data:
        for l in s:
            if l in cnt.keys():
                cnt[l] += 1
            else:
                cnt[l] = 1
    cnt = sorted(cnt.items(), key=lambda d: d[1], reverse=True)
    letter_dict = {}
    letter_cnt = 0
    for key, val in cnt[:threshold]:
        letter_dict[key] = letter_cnt
        letter_cnt += 1
    return letter_dict
    
def token_segmentation(data):
    
    punctuation = ['/', ':', ';', '+', '-', '*', '~', '|',
                   '[', ']', '{', '}', '(', ')',
                   '_', '=', '%', '&', '"', '`', '^', '—','\x92', 
                   '\xa0', '£', '«', '®', '´', 'º', 
                   '»', 'É', 'á', 'â', 'è', 'é',
                   'í', 'ï', 'ñ', 'ø', 'ú', 'ğ',
                   'ı', 'ĺ', 'ō', 'ד', 'ז', 
                   'ח', 'י', 'ם', 'מ', 'ק', 'ת', 
                   'ễ', '\u200b', '\u200e', '\u200f', 
                   '–', '—', '―', '‘', '’', '“', '”', '•', 
                   '…', '′', '‼', '€', '™', '↔', '●', '☀', 
                   '☁', '★', '☆', '☉', '☑', '☝', '☞', '☹', 
                   '☺', '♡', '♥', '⚾', '⛳', '✅', '✈', 
                   '✊', '✌', '✔', '✨', '❌', '❤', '➡', 
                   '⬅', '⬇', '《', '️', 'Ｒ', 'Ｔ', '🇦', 
                   '🇨', '🇪', '🇮', '🇱', '🇴', '🇵', 
                   '🇸', '🇺', '🌍', '🌏', '🌚', '🌹', 
                   '🍑', '🍷', '🍻', '🎈', '🎉', '🎧', 
                   '🎬', '🎾', '🏈', '🏢', '🏫', '🏻', 
                   '🏼', '🏽', '🐘', '👀', '👆', '👈', 
                   '👉', '👊', '👋', '👌', '👍', '👎', 
                   '👏', '👑', '👔', '👗', '👚', '👢', 
                   '👿', '💁', '💃', '💋', '💔', '💕', 
                   '💗', '💘', '💙', '💚', '💜', '💞', 
                   '💤', '💥', '💦', '💨', '💪', '💯', 
                   '💰', '📖', '📷', '📺', '🔅', '🔥', 
                   '🗽', '😁', '😂', '😃', '😄', '😅', 
                   '😆', '😇', '😉', '😊', '😍', '😎', 
                   '😑', '😒', '😓', '😔', '😘', '😜', 
                   '😝', '😡', '😢', '😣', '😩', '😬', 
                   '😰', '😱', '😳', '😴', '🙅', '🙌', 
                   '🙏', '🚀', '🚂', '🚨', '🤔', '🤖', '\U0010fc00','\\','$']


    res = [None]*len(data)
    for i, sent in enumerate(data):
        tmp_token_seq = sent.split()
        
        tmp_sent = ""
        for t in range(len(tmp_token_seq)):
            if ("http" in tmp_token_seq[t] \
                or "https" in tmp_token_seq[t]):
                tmp_sent += ""
            elif "'" in tmp_token_seq[t]:
                tmp_idx = tmp_token_seq[t].index("'")
                tmp_sent += " " + tmp_token_seq[t][:tmp_idx]
                tmp_sent += " " + tmp_token_seq[t][tmp_idx:]
            elif "’" in tmp_token_seq[t]:
                tmp_idx = tmp_token_seq[t].index("’")
                tmp_sent += " " + tmp_token_seq[t][:tmp_idx]
                tmp_sent += " " + tmp_token_seq[t][tmp_idx:]
            else:
                tmp_sent += " " + tmp_token_seq[t]

        #tmp_sent = tmp_sent.replace('\n', ' ').lower()
        for p in punctuation:
            tmp_sent = tmp_sent.replace(p, '')
                # tmp_sent = re.sub(r'\d+\.?\d*', ' NUM ', tmp_sent)
        
        res[i]=tmp_sent
    return res
            
def get_token_dict(data, show_up_threshold=10):
    
    cnt = {}
    for seq in data:
        for t in seq:
            if t in cnt.keys():
                cnt[t] += 1
            else:
                cnt[t] = 1
    cnt = sorted(cnt.items(), key=lambda d: d[1], reverse=True)
    
    ret = {}
    for i in cnt:
        if i[1] > show_up_threshold:
            ret[i[0]] = len(ret)
    return ret
    
    
if __name__ == "__main__":
    
    extract_zip_all("raw_trump", "ex_trump")
    data = read_json_text_all() # not preprocessed data

    ret = token_segmentation(data) # remove special characters from each tweet and add < at the end of each tweet ( and keep tweets seperated)

    all_data = ""# appended tweets ( not preprocessed)
    for tweet in data:
        all_data+=tweet

    book_chars = {uniq for uniq in all_data} # create letter dictionary 

   ### 1) Iterate through appended tweets with seq_length=25
    ## initialization
    rnn = RNN(book_chars=book_chars)
    seq_length=25
    X_chars= all_data[0:seq_length]; X = rnn.mapper[X_chars]
    Y_chars = all_data[1:seq_length+1]; Y = rnn.mapper[Y_chars]  
    h0 = np.zeros((100,1))
    loss = rnn.backward(X, Y, h0)
    ll = 0

    # Training loop    
    for k in range(8):
        for i in tqdm(range(0,len(all_data)-25,seq_length)):
            # select batch
            X_chars= all_data[i:i+seq_length]; X = rnn.mapper[X_chars]
            Y_chars = all_data[i+1:i+seq_length+1]; Y = rnn.mapper[Y_chars]

            # display loss every 250th update
            if np.mod(ll,100)==0:
                print(loss) 
            
            # write synthesized text in file every 500th update
            if np.mod(ll,500)==0:
                txt = rnn.predict(x=X[:,0], h=rnn.h, n=250)
                if np.mod(ll,1000)==0: 
                    with open('trump_synthezed/out_trumpp2.txt', 'a') as f:
                        print("\n*iter =*" +str(ll)+"*, smooth_loos=*"+str(loss)+"\n", file=f)
                        print("".join(rnn.word(txt)), file=f)
                print("**iter =**" +str(ll)+"**, smooth_loos=**"+str(loss)+"\n" )
                print("".join(rnn.word(txt)))
                
            # reset initial state if new epoche
            if i == 0:
                rnn.h=None

            # update loss
            loss = 0.999*loss + 0.001*rnn.backward(X, Y, h0); ll +=1

   ### 2) Consider each tweet as input(randomly choosen seq_length: 5<seq_length<tweet's_length)
    rnn = RNN(book_chars=book_chars)
    # initialization
    seq_length=25
    X_chars= all_data[0:seq_length]; X = rnn.mapper[X_chars]
    Y_chars = all_data[1:seq_length+1]; Y = rnn.mapper[Y_chars]    
    h0 = np.zeros((100,1))
    loss = rnn.backward(X, Y, h0)
    ll = 0

    #Training loop
    for k in range(8):
        for tweet in tqdm(ret):

            seq_length = random.randint(1, len(tweet)-1)
            if seq_length < 5: continue
            
            #select batch
            X_chars= tweet[0:seq_length]; X = rnn.mapper[X_chars]
            Y_chars = tweet[1:seq_length+1]; Y = rnn.mapper[Y_chars]       
            
            # display loss every 250th update
            if np.mod(ll,250)==0:
                print(loss) 
            
            # write synthesized text in file every 500th update
            if np.mod(ll,500)==0:
                txt = rnn.predict(x=X[:,0], h=rnn.h, n=250)
                if np.mod(ll,1000)==0: 
                    with open('trump_synthezed/out_trumpp3.txt', 'a') as f:
                        print("\n*iter =*" +str(ll)+"*, smooth_loos=*"+str(loss)+"\n", file=f)
                        print("".join(rnn.word(txt)), file=f)
                print("**iter =**" +str(ll)+"**, smooth_loos=**"+str(loss)+"\n" )
                print("".join(rnn.word(txt)))
                
            # reset init state for each tweet
            rnn.h=None
            loss = 0.999*loss + 0.001*rnn.backward(X, Y, h0); ll+=1

   ### 3) Train with seq_length=10 and consider tweets separately.
    rnn = RNN(book_chars=book_chars)
    #initialization
    seq_length=10
    X_chars= all_data[0:seq_length]; X = rnn.mapper[X_chars]
    Y_chars = all_data[1:seq_length+1]; Y = rnn.mapper[Y_chars] 
    h0 = np.zeros((100,1))
    loss = rnn.backward(X, Y, h0)
    ll = 0

    # training loop
    for k in range(3):
        for tweet in tqdm(ret):

            for iter in range(0,len(tweet),seq_length):
                if iter+seq_length+1>= len(tweet): continue

                #select batch    
                X_chars= tweet[iter:iter+seq_length]; X = rnn.mapper[X_chars]
                Y_chars = tweet[iter+1:iter+seq_length+1]; Y = rnn.mapper[Y_chars]       

                # display loss every 250th update
                if np.mod(ll,250)==0:
                    print(loss) 

                # write synthesized text in file every 500th update
                if np.mod(ll,500)==0:
                    txt = rnn.predict(x=X[:,0], h=rnn.h, n=250)
                    if np.mod(ll,1000)==0: 
                        with open('trump_synthezed/out_trumpp_length_10.txt', 'a') as f:
                            print("\n*iter =*" +str(ll)+"*, smooth_loos=*"+str(loss)+"\n", file=f)
                            print("".join(rnn.word(txt)), file=f)
                    print("**iter =**" +str(ll)+"**, smooth_loos=**"+str(loss)+"\n" )
                    print("".join(rnn.word(txt)))
                
                # reset init hidden state for each new tweet
                if iter==0: rnn.h=None
                loss = 0.999*loss + 0.001*rnn.backward(X, Y, h0); ll+=1