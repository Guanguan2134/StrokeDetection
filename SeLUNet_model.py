import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def SeLUNet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(inputs)
    conv1 = Conv2D(64, 3, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(pool1)
    conv2 = Conv2D(128, 3, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(pool2)
    conv3 = Conv2D(256, 3, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(pool3)
    conv4 = Conv2D(512, 3, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(conv4)
    drop4 = AlphaDropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(pool4)
    conv5 = Conv2D(1024, 3, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(conv5)
    drop5 = AlphaDropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(merge6)
    conv6 = Conv2D(512, 3, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(conv6)

    up7 = Conv2D(256, 2, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(merge7)
    conv7 = Conv2D(256, 3, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(conv7)

    up8 = Conv2D(128, 2, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(merge8)
    conv8 = Conv2D(128, 3, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(conv8)

    up9 = Conv2D(64, 2, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(merge9)
    conv9 = Conv2D(64, 3, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(conv9)
    conv9 = Conv2D(2, 3, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zero')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


