from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, LeakyReLU, UpSampling2D, Concatenate, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# U-Net
def UNet(pretrained_weights=None, activation='relu', input_size=(256,256,1)):
    inputs = Input(input_size)
    ENC, skip = encoder(inputs, activation=activation, order=[64, 128, 256, 512])

    # Bottle-neck
    convB5, _ = down_conv_block(ENC, kernel_num=1024, activation=activation, pool=False, drop=True)

    DEC = decoder(convB5, skip, activation=activation)
    if activation == 'leaky_relu':
        conv = Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(DEC)
        conv = LeakyReLU()(conv)
    else:
        conv = Conv2D(2, 3, activation=activation, padding='same', kernel_initializer='he_normal')(DEC)
    conv = Conv2D(1, 1, activation='sigmoid')(conv)

    model = Model(inputs=inputs, outputs=conv)

    model.compile(optimizer = Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

def down_conv_block(x, kernel_num=64, kernel_size=3, activation='relu', pool=True, drop=False):
    if activation == 'leaky_relu':
        conv = Conv2D(kernel_num, kernel_size, padding='same', kernel_initializer='he_normal')(x)
        conv = LeakyReLU()(conv)
        conv = Conv2D(kernel_num, kernel_size, padding='same', kernel_initializer='he_normal')(conv)
        conv = LeakyReLU()(conv)
    else:
        conv = Conv2D(kernel_num, kernel_size, activation=activation, padding='same', kernel_initializer='he_normal')(x)
        conv = Conv2D(kernel_num, kernel_size, activation=activation, padding='same', kernel_initializer='he_normal')(conv)

    if drop:
        conv = Dropout(0.5)(conv)
    if pool:
        out = MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        out = conv

    return out, conv

def up_conv_block(x1, x2, kernel_num=64, kernel_size=3, activation='relu'):
    if activation == 'leaky_relu':
        up = Conv2D(kernel_num, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(x1))
        up = LeakyReLU()(up)
        merge = concatenate([x2,up], axis=3)
        conv = Conv2D(kernel_num, kernel_size, padding='same', kernel_initializer='he_normal')(merge)
        conv = LeakyReLU()(conv)
        conv = Conv2D(kernel_num, kernel_size, padding='same', kernel_initializer='he_normal')(conv)
        out = LeakyReLU()(conv)
    else:
        up = Conv2D(kernel_num, 2, activation=activation, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(x1))
        merge = concatenate([x2,up], axis=3)
        conv = Conv2D(kernel_num, kernel_size, activation=activation, padding='same', kernel_initializer='he_normal')(merge)
        out = Conv2D(kernel_num, kernel_size, activation=activation, padding='same', kernel_initializer='he_normal')(conv)

    return out

def encoder(x, activation='relu', order=[64, 128, 256, 512]):
    skip = []
    for i in range(len(order)-1):
        x, conv = down_conv_block(x, kernel_num=order[i], activation=activation)
        skip.append(conv)
    x, conv = down_conv_block(x, kernel_num=order[-1], drop=True, activation=activation)
    skip.append(conv)

    return x, skip

def decoder(x, skip, activation='relu', order=[512, 256, 128, 64]):
    for i in range(len(order)):
        x = up_conv_block(x, skip[-(i+1)], kernel_num=order[i], activation=activation)
    return x

#################################################################################################################
# ResU-Net
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

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=Adam(lr=1e-4), loss=exp_dice_loss(exp=1.0), metrics=[dice_coef])

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

        dice = dice_coef(y_true, y_pred)
        #dice = generalized_dice(y_true, y_pred, exp)
        dice = K.clip(dice, K.epsilon(), 1 - K.epsilon())  # As log is used
        dice = K.pow(-K.log(dice), exp)
        if K.ndim(dice) == 2:
            dice = K.mean(dice, axis=-1)
        return dice

    return inner

def dice_coef(y_true, y_pred):
    """Computes Dice coefficients with additive smoothing.
    :param y_true: one-hot tensor multiplied by label weights (batch size, number of pixels, number of labels).
    :param y_pred: softmax probabilities, same shape as y_true. Each probability serves as a partial volume.
    :return: Dice coefficients (batch size, number of labels).
    """
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)