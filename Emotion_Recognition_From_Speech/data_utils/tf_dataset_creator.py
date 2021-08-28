import pickle
import tensorflow as tf

PATH_TO_SPLITTS = 'dataset/splitts/'
NAMES= {'train':0,'val':0,'test':0}

mfccs = None
labels = None

def _load_dat_in_memory(part):
    """[Loads the global variables mfccs and labels that are used from the generator gen()]

    Arguments:
        part {[string]} -- ['train', 'val' or 'test']
    """
    global mfccs, labels
    # train will be deleted once out of function
    sets = pickle.load(open(PATH_TO_SPLITTS+"splitted_mfcc_data_{}-{}.p".format(0.1,0.2), "rb"))
    # update the global variables
    for i in range(len(sets[part])):
        labels.append(sets[part][i]['emotion'])
        mfccs.append(sets[part][i]['mfcc'].T)
    mfccs = tf.keras.preprocessing.sequence.pad_sequences(mfccs)

def _gen():
    """[Generator for the tf dataset]

    Yields:
        [mfcc, label] -- [Yields a feature and its label (when you iterate with gen)]
    """
    for i in tf.range(0,len(labels)):
        yield mfccs[i], labels[i]

def get_dataset(name='train',**kwargs):
    """[Returns the tf dataset]

    Keyword Arguments:
        name {str} -- [description] (default: {'train'})

    Returns:
        [type] -- ['train', 'val' or 'test']
    """
    if name not in NAMES:
        print("ERROR, invalid value for name. name can be 'train', 'val' or 'test'.")
        exit()

    # initialize mfccs and labels
    global mfccs, labels
    mfccs = []
    labels = []
    
    _load_dat_in_memory(NAMES[name]) # prepare mffccs and labels for generator gen
    shapes = ((mfccs[0].shape[0],40),(8,))
    dataset = tf.data.Dataset.from_generator(_gen,
                                            output_types=(tf.float64, tf.int32),
                                            output_shapes=shapes)
    return dataset


## USAGE
def main():
    ds = get_dataset('test')

    for feature, lab in ds.take(6):
        print(feature.shape, lab)


if __name__ == "__main__":
    main()