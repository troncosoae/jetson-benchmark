import numpy as np
import argparse
import time
import os
import sys

from jtop_lib.JtopAdapter import JtopAdapter
from trt_lib.Interface import Interface as trtInterface
from importable_main import import_data, write_csv
from data_worker.data_worker import split_into_batches
from torch_lib.data_worker import suit4torch
from torch_lib.Interface import Interface as torchInterface
from torch_lib.Nets import LargeNet as torchLNet, \
    MediumNet as torchMNet, SmallNet as torchSNet


def get_args():

    parser = argparse.ArgumentParser(
        description="Mode selection through flags")
    parser.add_argument(
        '-eo', '--export_onnx', action="store_true",
        help="select export onnx mode", default=False)
    parser.add_argument(
        '-pth', '--path', help="path for onnx or trt file", type=str)
    parser.add_argument(
        '-s', '--size', help="net size", type=str)
    parser.add_argument(
        '-p', '--priority', help="execution priority", type=int, default=0)
    parser.add_argument(
        '-f', '--framework', help="select framework", type=str,
        default='torch')
    parser.add_argument(
        '-d', '--device', help="select device", type=str,
        default='cpu')
    parser.add_argument(
        '-bs', '--batch_size', help='select batch size', type=int, default=10)
    parser.add_argument(
        '-l', '--loops', help='loops in forward', type=int, default=1)
    parser.add_argument(
        '-ech', '--echo', action="store_true",
        help="select echo", default=False)
    args = parser.parse_args()
    return vars(args)


def export_onnx_main(path, size, framework, batch_size, **kwargs):

    X_data, Y_data = import_data()
    X, Y = suit4torch(X_data, Y_data)
    batches = split_into_batches(X, Y, batch_size)
    dummy_batch, _ = batches[0]

    net = torchSNet()
    if size == 'small_v1':
        net = torchSNet()
    elif size == 'medium_v1':
        net = torchMNet()
    elif size == 'large_v1':
        net = torchLNet()
    net_interface = torchInterface(net)
    net_interface.load_weights(
        f'saved_nets/saved_{framework}/{size}.pth')

    net_interface.convert2onnx(path, dummy_batch)


def suit4trt(batches, target_dtype=np.float32):
    new_batches = []
    for X_batch, Y_batch in batches:
        X = X_batch.float()
        Y = Y_batch.float()

        X = np.array(X, dtype=target_dtype)
        Y = np.array(Y, dtype=target_dtype)
        X = np.ascontiguousarray(X)
        # print(X, Y)
        new_batches.append((X, Y))
    return new_batches


def run_trt_main(batch_size, path, loops=1, priority=0, echo=False, **kwargs):

    target_dtype = np.float32

    X_data, Y_data = import_data()
    n_classes = int(np.max(Y_data) + 1)

    X, Y = suit4torch(X_data, Y_data)
    batches = split_into_batches(X, Y, batch_size)
    batches = suit4trt(batches)
    X_dummy, Y_dummy = batches[0]

    net_interface = trtInterface(
        path, X_dummy, batch_size=batch_size, n_classes=n_classes,
        target_dtype=target_dtype)

    batch_exec_times = []
    columns_bet = ['loop', 'batch', 'batch_time', 'time']

    if sys.platform == 'linux':
        os.nice(priority)

    with JtopAdapter(0.1) as tracker:

        initial_time = time.time()

        for loop in range(loops):
            if echo:
                print('loop')
            batch_count = 0
            for X_batch, Y_batch in batches:
                start_time = time.time()
                Y_pred = net_interface.predict_net(X_batch)
                batch_time = time.time() - start_time
                if echo:
                    print(f'batch_time: {batch_time:.8f}')
                batch_exec_times.append((
                    loop, batch_count, batch_time, time.time() - initial_time
                ))
                batch_count += 1

        procesor_tracked_values, columns_ptv = tracker.export_stats()

    name = path.split('/')[-1].split('.')[0]
    framework = name.split('_')[0]
    name = name.split('_')[1] + '_' + name.split('_')[2]
    write_csv(
        procesor_tracked_values, columns_ptv,
        f"performance_data/{'trt'}/{framework}/{'cuda'}_{priority}_" +
        f"{name}/tracked_values.csv")
    write_csv(
        batch_exec_times, columns_bet,
        f"performance_data/{'trt'}/{framework}/{'cuda'}_{priority}_" +
        f"{name}/batch_exec_times.csv")


if __name__ == "__main__":

    kwargs = get_args()

    if kwargs['export_onnx']:
        export_onnx_main(**kwargs)
    else:
        run_trt_main(**kwargs)
