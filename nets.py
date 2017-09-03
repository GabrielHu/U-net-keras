import keras
from keras import backend as K
from keras.layers import Input, Dropout, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras.models import Model

# change the loss function


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


'''
The UNET model is compiled in this function.
'''


def unet_model():
    inputs = Input((64, 64, 1))
    conv1 = Conv2D(filters=64, kernel_size=3,
                   activation='selu', padding='same', name='conv1_1', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(filters=64, kernel_size=3,
                   activation='selu', padding='same', name='conv1_2', kernel_initializer='he_normal')(conv1)
    conv1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filters=128, kernel_size=3,
                   activation='selu', padding='same', name='conv2_1', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(filters=128, kernel_size=3,
                   activation='selu', padding='same', name='conv2_2', kernel_initializer='he_normal')(conv2)
    conv2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filters=256, kernel_size=3,
                   activation='selu', padding='same', name='conv3_1', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(filters=256, kernel_size=3,
                   activation='selu', padding='same', name='conv3_2', kernel_initializer='he_normal')(conv3)
    conv3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(filters=512, kernel_size=3,  activation='selu',
                   padding='same', name='conv4_1', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(filters=512, kernel_size=3,  activation='selu',
                   padding='same', name='conv4_2', kernel_initializer='he_normal')(conv4)
    conv4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(filters=1024, kernel_size=3,
                   activation='selu', padding='same', name='conv5_1', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(filters=1024,  kernel_size=3,
                   activation='selu', padding='same', name='conv5_2', kernel_initializer='he_normal')(conv5)
    conv5 = Dropout(0.5)(conv5)

    up6 = concatenate([Conv2D(filters=512, kernel_size=3, activation='selu', padding='same',
                              name='deconv6', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5)), conv4])
    conv6 = Conv2D(filters=512, kernel_size=3,
                   activation='selu', padding='same', name='conv6_1', kernel_initializer='he_normal')(up6)
    #conv6 = Dropout(0.5)(conv6)
    conv6 = Conv2D(filters=512, kernel_size=3,
                   activation='selu', padding='same', name='conv6_2', kernel_initializer='he_normal')(conv6)

    up7 = concatenate([Conv2D(filters=256, kernel_size=3, activation='selu', padding='same',
                              name='deconv7', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6)), conv3])
    conv7 = Conv2D(filters=256, kernel_size=3,
                   activation='selu', padding='same', name='conv7_1', kernel_initializer='he_normal')(up7)
    #conv7 = Dropout(0.5)(conv7)
    conv7 = Conv2D(filters=256, kernel_size=3,
                   activation='selu', padding='same', name='conv7_2', kernel_initializer='he_normal')(conv7)

    up8 = concatenate([Conv2D(filters=128, kernel_size=3, activation='selu', padding='same',
                              name='deconv8', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7)), conv2])
    conv8 = Conv2D(filters=128,  kernel_size=3,
                   activation='selu', padding='same', name='conv8_1', kernel_initializer='he_normal')(up8)
    #conv8 = Dropout(0.5)(conv8)
    conv8 = Conv2D(filters=128, kernel_size=3,
                   activation='selu', padding='same', name='conv8_2', kernel_initializer='he_normal')(conv8)

    up9 = concatenate([Conv2D(filters=64, kernel_size=3, activation='selu', padding='same',
                              name='deconv9', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8)), conv1])
    conv9 = Conv2D(filters=64, kernel_size=3,
                   activation='selu', padding='same', name='conv9_1', kernel_initializer='he_normal')(up9)
    #conv9 = Dropout(0.5)(conv9)
    conv9 = Conv2D(filters=64, kernel_size=3,
                   activation='selu', padding='same', name='conv9_2', kernel_initializer='he_normal')(conv9)
    #conv9 = Dropout(0.5)(conv9)
    conv9 = Conv2D(filters=2, kernel_size=3, activation='selu',
                   padding='same', name='conv9_3', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(filters=1, kernel_size=1,
                    activation='sigmoid', name='conv10_1')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.summary()
    model.compile(optimizer=Adam(lr=1e-5),
                  loss='binary_crossentropy', metrics=[dice_coef])

    return model
