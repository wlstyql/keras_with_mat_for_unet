from __future__ import print_function
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import time
from data_mat import load_train_mat, load_test_mat, augmentation
import keras.backend as K
import scipy.io

'''
Deep-Learning using keras, u-net, '.mat' file version
for train, this saves weights file
need train file, test file
'''

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def get_unet():

    kwargs = dict( kernel_size= (3,3), strides= (1,1),
                   activation='relu', padding='same',
                   kernel_initializer='glorot_uniform' )

    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(filter, **kwargs)(inputs)
    conv1 = Conv2D(filter, **kwargs, name = 'conv1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filter*2, **kwargs)(pool1)
    conv2 = Conv2D(filter*2, **kwargs, name = 'conv2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filter*4, **kwargs)(pool2)
    conv3 = Conv2D(filter*4, **kwargs, name = 'conv3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    drop1 = Dropout(rate = 0.4, name='drop1')(pool3)

    conv4 = Conv2D(filter*8, **kwargs)(drop1)
    conv4 = Conv2D(filter*8, **kwargs, name = 'conv4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop2 = Dropout(rate = 0.4, name='drop2')(pool4)

    conv5 = Conv2D(filter*16, **kwargs)(drop2)
    conv5 = Conv2D(filter*16, **kwargs)(conv5)
    conv5 = Conv2D(filter*16, **kwargs, name='conv5')(conv5)
    drop3 = Dropout(rate=0.4, name='drop3')(conv5)

    up6 = concatenate([Conv2DTranspose(filter*8, (2, 2), strides=(2, 2), padding='same', kernel_initializer='glorot_uniform')(drop3), conv4], axis=3)
    conv6 = Conv2D(filter*8, **kwargs)(up6)
    conv6 = Conv2D(filter*8, **kwargs, name='conv6')(conv6)

    up7 = concatenate([Conv2DTranspose(filter*4, (2, 2), strides=(2, 2), padding='same', kernel_initializer='glorot_uniform')(conv6), conv3], axis=3)
    conv7 = Conv2D(filter*4, **kwargs)(up7)
    conv7 = Conv2D(filter*4, **kwargs, name='conv7')(conv7)

    up8 = concatenate([Conv2DTranspose(filter*2, (2, 2), strides=(2, 2), padding='same', kernel_initializer='glorot_uniform')(conv7), conv2], axis=3)
    conv8 = Conv2D(filter*2, **kwargs)(up8)
    conv8 = Conv2D(filter*2, **kwargs, name = 'conv8')(conv8)

    up9 = concatenate([Conv2DTranspose(filter,(2, 2), strides=(2, 2), padding='same', kernel_initializer='glorot_uniform')(conv8), conv1], axis=3)
    conv9 = Conv2D(filter, **kwargs)(up9)
    conv9 = Conv2D(filter, **kwargs, name='conv9')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='conv10')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=learning_rate), loss=loss_func, metrics=[rmse])

    return model

def plot_result_diff(data1, data2, data3, sl1, sl2):

    # plot slice 1, 2 of data1, data2, data2-data3
    diff = data2 - data3

    plt.figure()
    plt.subplot(421), plt.axis('off'), plt.set_cmap('jet'),
    plt.imshow(data1[sl1, :, :,0]), plt.colorbar(orientation='vertical')
    plt.subplot(422), plt.axis('off'), plt.set_cmap('jet')
    plt.imshow(data1[sl2, :, :,0]), plt.colorbar(orientation='vertical')
    plt.subplot(423), plt.axis('off'), plt.set_cmap('jet')
    plt.imshow(data2[sl1, :, :,0]), plt.colorbar(orientation='vertical')
    plt.subplot(424), plt.axis('off'), plt.set_cmap('jet')
    plt.imshow(data2[sl2, :, :,0]), plt.colorbar(orientation='vertical')
    plt.subplot(425), plt.axis('off'), plt.set_cmap('jet')
    plt.imshow(data3[sl1, :, :,0]), plt.colorbar(orientation='vertical')
    plt.subplot(426), plt.axis('off'), plt.set_cmap('jet')
    plt.imshow(data3[sl2, :, :,0]), plt.colorbar(orientation='vertical')
    plt.subplot(427), plt.axis('off'), plt.set_cmap('jet')
    plt.imshow(diff[sl1, :, :,0]), plt.colorbar(orientation='vertical')
    plt.subplot(428), plt.axis('off'), plt.set_cmap('jet')
    plt.imshow(diff[sl2, :, :,0]), plt.colorbar(orientation='vertical')

    # save figure with fname
    plt.savefig('%s.png' % fname, dpi=300)
    plt.show()

def train_and_predict():

    print('-'*30)
    print('Loading and augment train data...')
    print('-'*30)

    # load train, test '.mat' file from data_mat
    imgs_train, imgs_label = load_train_mat(train_mat_file)
    imgs_test, test_label = load_test_mat(test_mat_file)

    # load augmentation from data_mat
    train_aug, label_aug = augmentation(imgs_train, imgs_label, flip_ud=True, flip_lr=True, rot_90 =True,
                                         percent=0.5, augument_multiple=3) # 원래는 3

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    # save initial weights (Xavier initialization)
    model.save('weights_%s.h5' % fname)
    # training model checkpoint
    model_checkpoint = ModelCheckpoint('weights_%s.h5' % fname, monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    # model fitting, make training history
    history = model.fit(train_aug, label_aug, batch_size= batch, epochs=epoch, verbose=1, shuffle=True,
              validation_split=0.1, callbacks=[model_checkpoint])

    model.summary()
    end_time = time.time()
    elapsed_time = time.time() - start_time
    print('Training elapsed time : ', elapsed_time, ' sec')

    test_time = time.time()

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    # load pre-trained weights ( 'weights_' + fname )
    model.load_weights('weights_%s.h5' % fname)

    print('-'*30)
    print('Predicting on test data... ')
    print('-'*30)

    # prediction result data initialization
    imgs_result = model.predict(imgs_test, verbose=1)

    test_end_time = time.time()
    test_elapsed_time = test_end_time - test_time
    print('Test elapsed time : ', test_elapsed_time, ' sec')

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)

    # prediction result data save for '.mat' file
    scipy.io.savemat(fname + '.mat', {'imgs_result': imgs_result})

    plot_result_diff(imgs_test, test_label, imgs_result, 0, imgs_test.shape[0]-1)

    # plot and save history of rmse, loss
    plt.plot(history.history['rmse'])
    plt.title('RMSE')
    plt.savefig('%s_rmse.png' % fname)
    plt.show()

    plt.plot(history.history['loss'])
    plt.title('loss')
    plt.savefig('%s_loss.png' % fname)
    plt.show()
    print(fname)

    # root mean sqaure error result between label and prediction result
    rmse_result = np.sqrt(np.mean(np.square(test_label - imgs_result)))
    print('%s' % rmse_result)


if __name__ == '__main__':
    start_time = time.time()

    # learning parameters
    img_rows = 256
    img_cols = 256
    filter = 64
    batch = 16
    epoch = 100
    learning_rate = 1e-04

    # insert train, test mat file name
    train_mat_file = 'train_file_name'
    test_mat_file = 'test_file_name'
    loss_func = 'binary_crossentropy'
    fname = '%s_batch%s_e%s_%s' % (learning_rate, batch, epoch, train_file)

    # Deep-Learning start
    train_and_predict()