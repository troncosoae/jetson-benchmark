import numpy as np
from threading import Thread
import time
import psutil
import subprocess
import pandas as pd
import torch
import sys
import os
import csv

from torch._C import float32
if sys.platform == 'linux':
    from jtop import jtop

from data_worker.data_worker import unpickle, unpack_data, \
    combine_batches, split_into_batches
from tf_lib.data_worker import suit4tf
from tf_lib.Interface import Interface as tfInterface
from tf_lib.Nets import LargeNet as tfLNet, \
    MediumNet as tfMNet, SmallNet as tfSNet


class DummyJtop(Thread):
    def __init__(self, interval=0.5):
        self.Ts = interval
        self.cpu = {
            'CPU1': {'val': 1},
            'CPU2': {'val': 1},
            'CPU3': {'val': 1},
            'CPU4': {'val': 1}
        }
        self.gpu = {'val': 1}
        self.ram = {'use': 1, 'tot': 1}

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        pass


class JtopAdapter(Thread):
    def __init__(self, interval):
        self.interval = interval
        self.start_time = None
        self.jtop_inst = DummyJtop(interval=interval)
        if sys.platform == 'linux':
            self.jtop_inst = jtop(interval=interval)
        self.values = []
        self.stopped = True
        super().__init__()

    def __enter__(self):
        self.stopped = False
        self.start_time = time.time()
        self.jtop_inst.__enter__()
        self.start()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.stopped = True
        self.jtop_inst.__exit__(exception_type, exception_value, traceback)

    def read_stats(self):
        self.values.append((
            self.jtop_inst.cpu['CPU1']['val'],
            self.jtop_inst.cpu['CPU2']['val'],
            self.jtop_inst.cpu['CPU3']['val'],
            self.jtop_inst.cpu['CPU4']['val'],
            self.jtop_inst.gpu['val'],
            self.jtop_inst.ram['use']/self.jtop_inst.ram['tot'],
            time.time() - self.start_time
        ))

    def run(self):
        while not self.stopped:
            self.read_stats()
            time.sleep(self.interval)

    def export_stats(self):
        return self.values, ['CPU1', 'CPU2', 'CPU3', 'CPU4', 'GPU', 'RAM']


def execute_net_tf(
        net_interface, X_data, Y_data, batch_size, priority, loops=1,
        device='cuda', echo=True):

    if sys.platform == 'linux':
        os.nice(priority)

    if device not in ['cuda', 'cpu']:
        raise Exception("'device' parameter must be one of 'cuda' or 'cpu'")
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    batches = split_into_batches(X_data, Y_data, batch_size)

    batch_exec_times = []
    columns_bet = ['loop', 'batch', 'batch_time', 'time']

    with JtopAdapter(0.1) as tracker:

        initial_time = time.time()

        for loop in range(loops):
            if echo:
                print('loop:', loop)
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

    return procesor_tracked_values, columns_ptv, batch_exec_times, columns_bet


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


def run_forward_test(
        X_data, Y_data, net_size, saved_net_path, priority, device, framework):

    if device not in ['cuda', 'cpu']:
        raise Exception("'device' parameter must be one of 'cuda' or 'cpu'")
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if framework not in ['torch', 'tf']:
        raise Exception("'framework' parameter must be one of 'torch' or 'tf'")

    if framework == 'tf':
        net = tfSNet
        if net_size == 's':
            net = tfSNet
        elif net_size == 'm':
            net = tfMNet
        elif net_size == 'l':
            net = tfLNet

        print(net)

        net_interface = tfInterface(net)
        net_interface.load_weights(
            f'saved_nets/saved_{framework}/{saved_net_path}.pth')

        X, Y = suit4tf(X_data, Y_data)

        exec_batch_size = 10
        tracked_values, cols1, batch_exec_times, cols2 = execute_net_tf(
            net_interface, X, Y, exec_batch_size, priority,
            device=device, echo=False)

        write_csv(
            tracked_values, cols1,
            f"performance_data/{framework}/{device}_{priority}_" +
            f"{saved_net_path}/tracked_values.csv")
        write_csv(
            batch_exec_times, cols2,
            f"performance_data/{framework}/{device}_{priority}_" +
            f"{saved_net_path}/batch_exec_times.csv")


def write_csv(list, columns, path):
    with open(path, 'w+', newline='') as file:
        write = csv.writer(file)
        write.writerow(columns)
        write.writerows(list)


if __name__ == "__main__":

    X, Y = import_data()

    sizes = ['large_v1', 'medium_v1', 'small_v1']
    priorities = [-15, 0, 15]
    devices = ['cuda', 'cpu']

    for size in sizes:
        for priority in priorities:
            for device in devices:
                print(size, priority, device)
                run_forward_test(
                    X, Y, size[0], size, priority, device, 'tf')
