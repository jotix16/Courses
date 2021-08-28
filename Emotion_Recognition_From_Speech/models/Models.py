import tensorflow as tf
from tensorflow.keras import layers


def DataFlair(target_class_no=4 ,Units=300, input_shape=180):
    
    Model = tf.keras.Sequential([
        layers.Dense(Units,input_shape=[input_shape], kernel_initializer="he_normal", activation="relu"),
        layers.Dense(target_class_no,activation='softmax')])

    #optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=Decay)
    Model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
            
    Model.summary()
    return Model
    
def CNN2D(input_shape, target_class_no):
    def LSTM():
        return layers.LSTM(256)

    def LFLB2D(nr_filters, pool, input_shape=None):
        lflb = tf.keras.Sequential()

        if input_shape is None:
            lflb.add(layers.Conv2D(nr_filters,kernel_size=(3,3),padding='same'))
        else:
            lflb.add(layers.Conv2D(nr_filters,kernel_size=(3,3), input_shape=input_shape,padding='same'))

        lflb.add(layers.BatchNormalization())
        lflb.add(layers.Activation(activation = tf.keras.activations.elu))
        lflb.add(layers.MaxPool2D(pool_size=(pool,pool), strides=(pool,pool)))

        return lflb

    filters = [16, 32, 64, 128]
    pools = [2, 4, 4, 4]

    model = tf.keras.Sequential()
    model.add(LFLB2D(nr_filters=filters[0], pool=pools[0] , input_shape=input_shape))

    for nr_filters, pool in zip(filters[2:], pools[2:]):
        model.add(LFLB2D(nr_filters=nr_filters, pool=pool))

    model.add( layers.Flatten() )
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dense(target_class_no))

    opt = tf.keras.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
    #opt = tf.keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
                optimizer=opt,
                metrics=['accuracy'])
    return model

def CNN2D_LSTM(input_shape, target_class_no):
    def LSTM():
        return layers.LSTM(256)

    def LFLB2D(nr_filters, pool, input_shape=None):
        lflb = tf.keras.Sequential()

        if input_shape is None:
            lflb.add(layers.Conv2D(nr_filters,kernel_size=(3,3),padding='same'))
        else:
            lflb.add(layers.Conv2D(nr_filters,kernel_size=(3,3), input_shape=input_shape,padding='same'))

        lflb.add(layers.BatchNormalization())
        lflb.add(layers.Activation(activation = tf.keras.activations.elu))
        lflb.add(layers.MaxPool2D(pool_size=(pool,pool), strides=(pool,pool)))

        return lflb

    filters = [64, 64, 128, 128]
    pools = [2, 4, 4, 4]

    model = tf.keras.Sequential()
    model.add(LFLB2D(nr_filters=filters[0], pool=pools[0] , input_shape=input_shape))

    for nr_filters, pool in zip(filters[2:], pools[2:]):
        model.add(LFLB2D(nr_filters=nr_filters, pool=pool))

    model.add( layers.Reshape((-1,128)) )
    model.add(LSTM())
    model.add(layers.Dense(target_class_no))
    model.summary()

    opt = tf.keras.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
    #opt = tf.keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
                optimizer=opt,
                metrics=['accuracy'])
    return model

def CNN1D_LSTM_AUDIO(input_shape, target_class_no):
    def LFLB1D(input_shape=None, kernel=None):
        lflb = tf.keras.Sequential()

        if input_shape is None:
            lflb.add(layers.Conv1D(kernel,kernel_size=4, padding='same'))
        else:
            lflb.add(layers.Conv1D(kernel,kernel_size=4, input_shape=input_shape ,padding='same'))

        lflb.add(layers.BatchNormalization())
        lflb.add(layers.Activation(activation=tf.keras.activations.relu))
        lflb.add(layers.MaxPool1D(pool_size=4, strides=4))

        return lflb

    def LSTM():
        return layers.LSTM(512)


    filters = [32, 64, 128, 256]
    audio_model = tf.keras.Sequential()
    audio_model.add(LFLB1D(input_shape, filters[0]))

    for i in range(3):
        audio_model.add(LFLB1D(kernel=filters[i+1]))

    audio_model.add( layers.Reshape((-1,256)) )
    audio_model.add(LSTM())
    audio_model.add(layers.Dropout(0.25))

    audio_model.add(layers.Dense(target_class_no))
    audio_model.summary()

    opt = tf.keras.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
    #opt = tf.keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
    audio_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
                optimizer=opt,
                metrics=['accuracy'])

    return audio_model

