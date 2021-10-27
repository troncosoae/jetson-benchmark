import csv

from data_worker.data_worker import unpickle, unpack_data, \
    combine_batches


def import_data():
    batches_names = [
        'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
        'data_batch_5'
    ]

    data_batches = [
        unpickle(f'datasets/cifar-10-batches-py/{batch_name}') for batch_name
        in batches_names]

    unpacked_batches = [
        (unpack_data(data_batch)) for data_batch
        in data_batches]

    X, Y = combine_batches(unpacked_batches)

    return X, Y


def write_csv(list, columns, path):
    with open(path, 'w+', newline='') as file:
        write = csv.writer(file)
        write.writerow(columns)
        write.writerows(list)
