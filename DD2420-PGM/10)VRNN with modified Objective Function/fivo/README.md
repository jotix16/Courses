# Filtering Variational Objectives

This folder contains a TensorFlow implementation of the algorithms from

Chris J. Maddison\*, Dieterich Lawson\*, George Tucker\*, Nicolas Heess, Mohammad Norouzi, Andriy Mnih, Arnaud Doucet, and Yee Whye Teh. "Filtering Variational Objectives." NIPS 2017.

[https://arxiv.org/abs/1705.09279](https://arxiv.org/abs/1705.09279)

This code implements 3 different bounds for training sequential latent variable models: the evidence lower bound (ELBO), the importance weighted auto-encoder bound (IWAE), and the filtering variational objective (FIVO).

Additionally it contains an implementation of the variational recurrent neural network (VRNN), a sequential latent variable model that can be trained using these three objectives. This repo provides code for training a VRNN to do sequence modeling of pianoroll.

#### Directory Structure
The important parts of the code are organized as follows.

```
fivo.py           # main script, contains flag definitions
runners.py        # graph construction code for training and evaluation
bounds.py         # code for computing each bound
data
├── datasets.py                    # readers for pianoroll and speech datasets
├── calculate_pianoroll_mean.py    # preprocesses the pianoroll datasets
└── create_timit_dataset.py        # preprocesses the TIMIT dataset
models
└── vrnn.py       # variational RNN implementation
bin
├── run_train.sh            # an example script that runs training
├── run_eval.sh             # an example script that runs evaluation
└── download_pianorolls.sh  # a script that downloads the pianoroll files
```

### Training on Pianorolls

Requirements before we start:

* TensorFlow (see [tensorflow.org](http://tensorflow.org) for how to install)
* [scipy](https://www.scipy.org/)
* [sonnet](https://github.com/deepmind/sonnet)


#### Download the Data

The pianoroll datasets are encoded as pickled sparse arrays and are available at [http://www-etud.iro.umontreal.ca/~boulanni/icml2012](http://www-etud.iro.umontreal.ca/~boulanni/icml2012). You can use the script `bin/download_pianorolls.sh` to download the files into a directory of your choosing.
```
export PIANOROLL_DIR=~/pianorolls
mkdir $PIANOROLL_DIR
sh bin/download_pianorolls.sh $PIANOROLL_DIR
```

#### Preprocess the Data

The script `calculate_pianoroll_mean.py` loads a pianoroll pickle file, calculates the mean, updates the pickle file to include the mean under the key `train_mean`, and writes the file back to disk in-place. You should do this for all pianoroll datasets you wish to train on.

```
python data/calculate_pianoroll_mean.py --in_file=$PIANOROLL_DIR/piano-midi.de.pkl
python data/calculate_pianoroll_mean.py --in_file=$PIANOROLL_DIR/nottingham.de.pkl
python data/calculate_pianoroll_mean.py --in_file=$PIANOROLL_DIR/musedata.pkl
python data/calculate_pianoroll_mean.py --in_file=$PIANOROLL_DIR/jsb.pkl
```

#### Training 
This is very similar to training on pianoroll datasets, with just a few flags switched.
```
python fivo.py \
--mode=train \
--logdir=/tmp/fivo-jsb \
--model=vrnn \
--latent_size=32 \
--bound=fivo \
--summarize_every=50 \
--batch_size=4 \
--num_samples=4 \
--learning_rate=0.0003 \
--max_steps=5000 \
--dataset_path="$PIANOROLL_DIR/jsb.pkl" \
--dataset_type="pianoroll"
```
#### Evaluation
This is very similar to training on pianoroll datasets, with just a few flags switched.
```
python fivo.py \
--mode=eval \
--split=test \
--bound=iwae \
--alsologtostderr \
--logdir=/tmp/fivo-jsb \
--model=vrnn \
--latent_size=32 \
--batch_size=4 \
--num_samples=4 \
--dataset_path="$PIANOROLL_DIR/jsb.pkl" \
--dataset_type="pianoroll"
```

### Tensorboard
```
tensorboard --logdir=/tmp/fivo-jsb/
```
One has to move or delete created files in /tmp/fivo-jsb/ before training or evaluating again.
