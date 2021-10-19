import pickle
import numpy as np
import matplotlib.pyplot as plt


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def unpack_data(data_dict):
    N = len(data_dict[b'labels'])
    Y = np.array(data_dict[b'labels'])
    data = data_dict[b'data']

    X = np.zeros((N, 32, 32, 3), dtype=data.dtype)

    for i in range(N):
        data_i = data[i, :]
        temp = data_i.reshape(3, 32, 32)
        X[i, :, :, :] = np.moveaxis(temp, 0, -1)

    return X, Y


def combine_batches(batches):
    N = 0
    img_shape = batches[0][0].shape[1:4]
    print(img_shape)
    for Xi, _ in batches:
        N += Xi.shape[0]
        if Xi.shape[1:4] != img_shape:
            raise Exception('databatches must have same shape')

    X = np.zeros((N, *img_shape))
    Y = np.zeros(N)

    last_n = 0
    for Xi, Yi in batches:
        n = Xi.shape[0]
        X[last_n:last_n+n, :, :, :] = Xi
        Y[last_n:last_n+n] = Yi
        last_n += n

    return X, Y


def split_into_batches(X, Y, batch_size):
    N = X.shape[0]
    batch_count = int(N / batch_size)
    batch_count += 1 if N % batch_size > 0 else 0

    batches = []

    start = 0
    while start < N:
        end = start + batch_size if start + batch_size <= N else N
        Xi = X[start:end, :, :, :]
        Yi = Y[start:end]
        start = end
        batches.append((Xi, Yi))

    return batches


def display_img(array):
    plt.imshow(array)
    plt.show()
