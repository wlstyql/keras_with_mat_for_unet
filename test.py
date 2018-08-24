from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time
from train_ori import get_unet, plot_result_diff
from data_mat import load_train_mat, load_test_mat, augumentation, load_images_from_folder
import keras.backend as K
import scipy.io
import os

'''
for test, this doesn't save weights file,
but load weights file
'''

if __name__ == '__main__':

    K.set_image_data_format('channels_last')  # TF dimension ordering in this code

    # insert test mat file name
    test_mat_file = 'test_file_name'
    # insert weights load name wo/ weights_
    fname = 'result_file_name'
    start_time = time.time()

    print('-'*30)
    print('Load model...')
    print('-'*30)
    # load u-net model from train
    model = get_unet()
    model.summary()

    test_time = time.time()
    print('-'*30)
    print('Loading test data...')

    # load test '.mat' file from data_mat
    imgs_test, test_label = load_test_mat(test_mat_file)

    # load pre-trained weights ( 'weights_' + fname )
    model.load_weights('weights_%s.h5' % fname)

    # prediction result data initialization
    imgs_result = np.zeros((imgs_test.shape), dtype='float32')

    test_end_time = time.time()
    test_elapsed_time = test_end_time - test_time
    print('Test elapsed time : ', test_elapsed_time, ' sec')

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)

    # prediction result data for '.mat' file
    scipy.io.savemat(fname + '_test.mat', {'imgs_result': imgs_result})
    print(fname)

    # root mean sqaure error result between label and prediction result
    rmse_result = np.sqrt(np.mean(np.square(test_label - imgs_result)))
    print('%s' % rmse_result)