# Emotion Recognition from Speech

## Requirements.txt
Install the depencies by running:
```
bash Requirements.txt
```

#### Directory Structure
The important parts of the code are organized as follows.

```
main.py                     # main script

data_utils
├── data_loader.py          # reader for the speech datasets
├── data_preprocessor.py    # preprocessor which extracts features from the samples
└── tf_data_generator.py    # KERAS/TF data generator

models
└── LSTM_based.py           # LSTM based model(Only an example, doesnt exist)

```


## Code style
Coding style is not that important at this point. But make sure you add an example inf form of an main() function or comment for the implemented parts.

## Running
In main.py we keep the progresss towards the endgoal. All implementations take place outside and we only import them in main.py.

