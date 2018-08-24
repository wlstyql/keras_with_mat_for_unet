import scipy.io
import numpy as np
import matplotlib.pyplot as plt

'''
data set input for '.mat' file of matlab
data sets must have [slice, row, col] 3D data
'''

def load_train_mat(name):

    # load training mat file
    tmpdata = scipy.io.loadmat(name)
    # check variable name for train(3D) in your mat file
    train = tmpdata['variable_name1']

    train = np.array(train, dtype='float32')
    # check variable name for train label(3D) in your mat file
    label = tmpdata['variable_name2']
    label = np.array(label, dtype='float32')

    return train, label

def load_test_mat(name):

    # load test mat file
    tmpdata2 = scipy.io.loadmat(name)
    # check variable name for test(3D) in your mat file
    test = tmpdata2['variable_name1']
    test = np.array(test, dtype='float32')
    # for tensor
    test = test[..., np.newaxis]

    test_label = tmpdata2['variable_name2']
    # check variable name for test label(3D) in your mat file
    test_label = np.array(test_label, dtype='float32')
    # for tensor
    test_label = test_label[..., np.newaxis]

    return test, test_label

def plot_figure(data1, data2, sl1, sl2, sl3):

    # plot slice 1, 2, 3 of data1(3D), data2(3D)
    plt.figure()
    plt.subplot(321), plt.axis('off'), plt.set_cmap('jet'),
    plt.imshow(data1[sl1, :, :]), plt.colorbar(orientation='vertical')
    plt.subplot(322), plt.axis('off'), plt.set_cmap('jet')
    plt.imshow(data2[sl1, :, :]), plt.colorbar(orientation='vertical')
    plt.subplot(323), plt.axis('off'), plt.set_cmap('jet')
    plt.imshow(data1[sl2, :, :]), plt.colorbar(orientation='vertical')
    plt.subplot(324), plt.axis('off'), plt.set_cmap('jet')
    plt.imshow(data2[sl2, :, :]), plt.colorbar(orientation='vertical')
    plt.subplot(325), plt.axis('off'), plt.set_cmap('jet')
    plt.imshow(data1[sl3, :, :]), plt.colorbar(orientation='vertical')
    plt.subplot(326), plt.axis('off'), plt.set_cmap('jet')
    plt.imshow(data2[sl3, :, :]), plt.colorbar(orientation='vertical')
    plt.show()

def augmentation(train, label, flip_ud, flip_lr, rot_90, percent, augment_multiple):

    # augmentation for provide overfitting
    train_out = np.zeros((train.shape[0]*int(augment_multiple),train.shape[1],train.shape[2]), dtype = 'float32')
    label_out = np.zeros((label.shape[0]*int(augment_multiple),label.shape[1],label.shape[2]), dtype = 'float32')

    for j in range(0, int(augment_multiple)):
        for i in range(0,train.shape[0]):

            train_sq = np.squeeze(train[i])
            label_sq = np.squeeze(label[i])

            if rot_90 == True:
                # random percent rotate 90
                if np.random.random() < percent:
                    train_sq = np.rot90(train_sq,1)
                    label_sq = np.rot90(label_sq,1)

            if flip_ud == True:
                # random percent flip up-down
                if np.random.random() < percent:
                    train_sq = np.flipud(train_sq)
                    label_sq = np.flipud(label_sq)

            if flip_lr == True :
                # random percent flip left-right
                if np.random.random() < percent:
                    train_sq = np.fliplr(train_sq)
                    label_sq = np.fliplr(label_sq)

            train_out[train.shape[0]*j+i] = train_sq
            label_out[label.shape[0]*j+i] = label_sq

    plot_figure(train_out, label_out, 0, train.shape[0], train.shape[0]*int(augment_multiple)-1)

    # for tensor
    train_out = train_out[...,np.newaxis]
    label_out = label_out[...,np.newaxis]

    return train_out, label_out

if __name__ == '__main__':

    # for mat file data check
    train, label= load_train_mat('test_file_name')
    plot_figure(train, label, 0, int(train.shape[0] / 2), train.shape[0]-1)
