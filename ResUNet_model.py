import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def ResUNet(pretrained_weights = None, input_size = (256,256,1)):
    f = [16, 32, 64, 128, 256]

    ## Encoder
    inputs = Input(input_size)
    e1 = stem(inputs, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)

    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)

    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])

    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])

    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])

    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])

    outputs = Conv2D(1, 1, activation='sigmoid')(d4)

    model = Model(input = inputs, output = outputs)

    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def bn_act(x, act=True):
    x = BatchNormalization()(x)
    if act == True:
        x = Activation("relu")(x)
    return x


def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv


def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = Add()([shortcut, res])
    return output


def upsample_concat_block(x, xskip):
    u = UpSampling2D((2, 2))(x)
    c = Concatenate()([u, xskip])
    return c

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

####################################################################################################
def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, res_path])
    return res_path


def encoder(x):
    to_decoder = []

    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='relu')(main_path)

    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)

    shortcut = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1))(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = add([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [256, 256], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    return to_decoder


def decoder(x, from_encoder):
    main_path = UpSampling2D(size=(2, 2))(x)
    main_path = concatenate([main_path, from_encoder[2]], axis=3)
    main_path = res_block(main_path, [256, 256], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=3)
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=3)
    main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)])

    return main_path


def ResUNet2(input_shape=(256,256,1)):
    inputs = Input(shape=input_shape)

    to_decoder = encoder(inputs)

    path = res_block(to_decoder[2], [512, 512], [(2, 2), (1, 1)])

    path = decoder(path, from_encoder=to_decoder)

    path = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(path)

    model = Model(input=inputs, output=path)
    optimizer = Adadelta()

    model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])
    return model

#################################################################################################################
def ResUNet3(pretrained_weights = None, input_size = (256,256,1)):

    """ first encoder for spect image """
    input_seg = Input(input_size)
    input_segBN = BatchNormalization()(input_seg)

    conv1_spect = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_segBN)
    conv1_spect = BatchNormalization()(conv1_spect)
    conv1_spect = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_spect)
    conv1_spect = BatchNormalization(name='conv_spect_32')(conv1_spect)
    conv1_spect = Add()([conv1_spect, input_segBN])
    pool1_spect = MaxPool2D(pool_size=(2, 2))(conv1_spect)


    conv2_spect_in = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1_spect)
    conv2_spect_in = BatchNormalization()(conv2_spect_in)
    conv2_spect = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_spect_in)
    conv2_spect = BatchNormalization(name='conv_spect_64')(conv2_spect)
    conv2_spect = Add()([conv2_spect, conv2_spect_in])
    pool2_spect = MaxPool2D(pool_size=(2, 2))(conv2_spect)

    conv3_spect_in = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2_spect)
    conv3_spect_in = BatchNormalization()(conv3_spect_in)
    conv3_spect = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_spect_in)
    conv3_spect = BatchNormalization(name='conv_spect_128')(conv3_spect)
    conv3_spect = Add()([conv3_spect, conv3_spect_in])
    pool3_spect = MaxPool2D(pool_size=(2, 2))(conv3_spect)

    conv4_spect_in = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3_spect)
    conv4_spect_in = BatchNormalization()(conv4_spect_in)
    conv4_spect = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_spect_in)
    conv4_spect = BatchNormalization(name='conv_spect_256')(conv4_spect)
    conv4_spect = Add()([conv4_spect, conv4_spect_in])
    drop4_spect = Dropout(0.5)(conv4_spect)
    pool4_spect = MaxPool2D(pool_size=(2, 2))(drop4_spect)

    """ second encoder for ct image """
    up7_cm = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(pool4_spect)) #24x24
    up7_cm = BatchNormalization()(up7_cm)
    merge7_cm = concatenate([drop4_spect, up7_cm], axis=3)  # cm: cross modality
    merge7_cm = BatchNormalization()(merge7_cm)
    conv7_cm = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7_cm)
    conv7_cm_in = BatchNormalization()(conv7_cm)
    conv7_cm = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7_cm_in)
    conv7_cm = BatchNormalization(name='decoder_conv_256')(conv7_cm)
    conv7_cm = Add()([conv7_cm, conv7_cm_in])

    up8_cm = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7_cm))
    up8_cm = BatchNormalization()(up8_cm)
    merge8_cm = concatenate([conv3_spect, up8_cm], axis=3)  # cm: cross modality
    merge8_cm = BatchNormalization()(merge8_cm)
    conv8_cm = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8_cm)
    conv8_cm_in = BatchNormalization()(conv8_cm)
    conv8_cm = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8_cm_in)
    conv8_cm = BatchNormalization(name='decoder_conv_128')(conv8_cm)
    conv8_cm = Add()([conv8_cm, conv8_cm_in])

    up9_cm = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8_cm))
    up9_cm = BatchNormalization()(up9_cm)
    merge9_cm = concatenate([conv2_spect, up9_cm], axis=3)  # cm: cross modality
    merge9_cm = BatchNormalization()(merge9_cm)
    conv9_cm = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9_cm)
    conv9_cm_in = BatchNormalization()(conv9_cm)
    conv9_cm = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_cm_in)
    conv9_cm = BatchNormalization(name='decoder_conv_64')(conv9_cm)
    conv9_cm = Add()([conv9_cm, conv9_cm_in])

    up10_cm = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv9_cm))
    up10_cm = BatchNormalization()(up10_cm)
    merge10_cm = concatenate([conv1_spect, up10_cm], axis=3)  # cm: cross modality
    merge10_cm = BatchNormalization()(merge10_cm)
    conv10_cm = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10_cm)
    conv10_cm_in = BatchNormalization()(conv10_cm)
    conv10_cm = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10_cm_in)
    conv10_cm = BatchNormalization(name='decoder_conv_32')(conv10_cm)
    conv10_cm = Add()([conv10_cm, conv10_cm_in])

    conv11_cm = Conv2D(filters=6, kernel_size=3, activation='relu', padding='same')(conv10_cm)
    conv11_cm = BatchNormalization()(conv11_cm)
    out = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same', name='segmentation')(conv11_cm)
    # if channels_first:
    #     new_shape = tuple(range(1, K.ndim(x)))
    #     new_shape = new_shape[1:] + new_shape[:1]
    #     x = Permute(new_shape)(x)

    # image_size = tuple((256, 256))

    # x = Reshape((np.prod(image_size), 3))(out)

    model = Model(inputs=input_seg, outputs=out)
    model.compile(optimizer=Adam(lr=1e-3), loss=exp_dice_loss(exp=1.0))
    # model.summary()

    return model

def exp_dice_loss(exp=1.0):
    """
    :param exp: exponent. 1.0 for no exponential effect, i.e. log Dice.
    """

    def inner(y_true, y_pred):
        """Computes the average exponential log Dice coefficients as the loss function.
        :param y_true: one-hot tensor multiplied by label weights, (batch size, number of pixels, number of labels).
        :param y_pred: softmax probabilities, same shape as y_true. Each probability serves as a partial volume.
        :return: average exponential log Dice coefficient.
        """

        dice = dice_coef2(y_true, y_pred)
        #dice = generalized_dice(y_true, y_pred, exp)
        dice = K.clip(dice, K.epsilon(), 1 - K.epsilon())  # As log is used
        dice = K.pow(-K.log(dice), exp)
        if K.ndim(dice) == 2:
            dice = K.mean(dice, axis=-1)
        return dice

    return inner

def dice_coef2(y_true, y_pred):
    """Computes Dice coefficients with additive smoothing.
    :param y_true: one-hot tensor multiplied by label weights (batch size, number of pixels, number of labels).
    :param y_pred: softmax probabilities, same shape as y_true. Each probability serves as a partial volume.
    :return: Dice coefficients (batch size, number of labels).
    """
    smooth = 1.0
    y_true = K.cast(K.not_equal(y_true, 0), K.floatx())  # Change to binary
    intersection = K.sum(y_true * y_pred, axis=1)  # (batch size, number of labels)
    union = K.sum(y_true + y_pred, axis=1)  # (batch size, number of labels)
    return (2. * intersection + smooth) / (union + smooth)  # (batch size, number of labels)