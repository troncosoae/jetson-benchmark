import numpy as np
from threading import Thread
import time
import psutil
import subprocess
import pandas as pd
import torch
import sys
import os

from data_worker.data_worker import unpickle, unpack_data, \
    combine_batches, split_into_batches
from torch_lib.data_worker import suit4torch
from torch_lib.Interface import Interface as torchInterface
from torch_lib.Nets import LargeNet as torchLNet, \
    MediumNet as torchMNet, SmallNet as torchSNet

from tf_lib.data_worker import suit4tf
from tf_lib.Interface import Interface as tfInterface
from tf_lib.Nets import LargeNet as tfLNet, \
    MediumNet as tfMNet, SmallNet as tfSNet


class GpuReader(Thread):

    def __init__(self):
        self.process = subprocess.Popen(['tegrastats'], stdout=subprocess.PIPE)
        self.stopped = True
        self.values = {}
        super().__init__()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.stop()

    def start(self):
        self.stopped = False
        super().start()

    def stop(self):
        self.stopped = True

    def run(self):
        while not self.stopped:
            resp = self.process.stdout.readline().strip().decode('utf-8')
            resp_array = resp.split(' ')
            idx = resp_array.index('GR3D_FREQ')
            self.values['GR3D_FRWQ'] = resp_array[idx + 1]


class CpuGpuTracker(Thread):

    def __init__(self, Ts):
        self.Ts = Ts
        self.stopped = True
        self.start_time = None
        self.values = []
        super().__init__()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.stop()

    def start(self):
        self.stopped = False
        self.start_time = time.time()
        super().start()

    def stop(self):
        self.stopped = True

    def run(self):
        while not self.stopped:
            mem = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            gpu_percent = 20
            self.values.append((
                    time.time() - self.start_time, mem.percent, cpu_percent,
                    gpu_percent
                ))
            time.sleep(self.Ts)

    def get_values(self):
        return pd.DataFrame(
            self.values, columns=[
                'time', 'memory', 'cpu_percent', 'gpu_percent']
            )


def execute_net_torch(
        net_interface, X_data, Y_data, batch_size, priority, loops=1,
        device='cuda', echo=True):

    if sys.platform == 'linux':
        os.nice(priority)

    if device not in ['cuda', 'cpu']:
        raise Exception("'device' parameter must be one of 'cuda' or 'cpu'")
    device = torch.device('cpu')
    if device == 'cuda':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if echo and not torch.cuda.is_available():
            print("'cuda' device is not available. Using 'cpu' instead.")

    batches = split_into_batches(X_data, Y_data, batch_size)

    batch_exec_times = []

    procesor_tracked_values = None
    with CpuGpuTracker(0.1) as tracker:

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

        procesor_tracked_values = tracker.get_values()

    batch_exec_times = pd.DataFrame(
        batch_exec_times, columns=['loop', 'batch_count', 'batch_time', 'time']
    )
    return procesor_tracked_values, batch_exec_times


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

    procesor_tracked_values = None
    with CpuGpuTracker(0.1) as tracker:

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

        procesor_tracked_values = tracker.get_values()

    batch_exec_times = pd.DataFrame(
        batch_exec_times, columns=['loop', 'batch_count', 'batch_time', 'time']
    )
    return procesor_tracked_values, batch_exec_times


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

    if framework not in ['torch', 'tf']:
        raise Exception("'framework' parameter must be one of 'torch' or 'tf'")

    if framework == 'torch':
        net = torchSNet()
        if net_size == 's':
            net = torchSNet()
        elif net_size == 'm':
            net = torchMNet()
        elif net_size == 'l':
            net = torchLNet()

        net_interface = torchInterface(net)
        net_interface.load_weights(
            f'saved_nets/saved_{framework}/{saved_net_path}.pth')

        X, Y = suit4torch(X_data, Y_data)

        exec_batch_size = 10
        tracked_values, batch_exec_times = execute_net_torch(
            net_interface, X, Y, exec_batch_size, priority,
            device=device, echo=False)

        tracked_values.to_csv(
            f"performance_data/{framework}/{saved_net_path}/" +
            "tracked_values.csv")
        batch_exec_times.to_csv(
            f"performance_data/{framework}/{saved_net_path}/" +
            "batch_exec_times.csv")
    elif framework == 'tf':
        net = tfSNet
        if net_size == 's':
            net = tfSNet
        elif net_size == 'm':
            net = tfMNet
        elif net_size == 'l':
            net = tfLNet

        net_interface = tfInterface(net)
        net_interface.load_weights(
            f'saved_nets/saved_{framework}/{saved_net_path}.pth')

        X, Y = suit4tf(X_data, Y_data)

        exec_batch_size = 10
        tracked_values, batch_exec_times = execute_net_tf(
            net_interface, X, Y, exec_batch_size, priority,
            device=device, echo=False)

        tracked_values.to_csv(
            f"performance_data/{framework}/{device}_{priority}_" +
            f"{saved_net_path}/batch_exec_times.csv")
        batch_exec_times.to_csv(
            f"performance_data/{framework}/{device}_{priority}_" +
            f"{saved_net_path}/batch_exec_times.csv")


if __name__ == "__main__":

    X, Y = import_data()

    run_forward_test(
        X, Y, 'l', 'large_v1', -15, 'cuda', 'tf')

    run_forward_test(
        X, Y, 'm', 'medium_v1', -15, 'cuda', 'tf')

    run_forward_test(
        X, Y, 's', 'small_v1', -15, 'cuda', 'tf')

    run_forward_test(
        X, Y, 'l', 'large_v1', 0, 'cuda', 'tf')

    run_forward_test(
        X, Y, 'm', 'medium_v1', 0, 'cuda', 'tf')

    run_forward_test(
        X, Y, 's', 'small_v1', 0, 'cuda', 'tf')

    run_forward_test(
        X, Y, 'l', 'large_v1', 15, 'cuda', 'tf')

    run_forward_test(
        X, Y, 'm', 'medium_v1', 15, 'cuda', 'tf')

    run_forward_test(
        X, Y, 's', 'small_v1', 15, 'cuda', 'tf')

    run_forward_test(
        X, Y, 'l', 'large_v1', -15, 'cpu', 'tf')

    run_forward_test(
        X, Y, 'm', 'medium_v1', -15, 'cpu', 'tf')

    run_forward_test(
        X, Y, 's', 'small_v1', -15, 'cpu', 'tf')

    run_forward_test(
        X, Y, 'l', 'large_v1', 0, 'cpu', 'tf')

    run_forward_test(
        X, Y, 'm', 'medium_v1', 0, 'cpu', 'tf')

    run_forward_test(
        X, Y, 's', 'small_v1', 0, 'cpu', 'tf')

    run_forward_test(
        X, Y, 'l', 'large_v1', 15, 'cpu', 'tf')

    run_forward_test(
        X, Y, 'm', 'medium_v1', 15, 'cpu', 'tf')

    run_forward_test(
        X, Y, 's', 'small_v1', 15, 'cpu', 'tf')

    run_forward_test(
        X, Y, 'l', 'large_v1', -15, 'cuda', 'torch')

    run_forward_test(
        X, Y, 'm', 'medium_v1', -15, 'cuda', 'torch')

    run_forward_test(
        X, Y, 's', 'small_v1', -15, 'cuda', 'torch')

    run_forward_test(
        X, Y, 'l', 'large_v1', 0, 'cuda', 'torch')

    run_forward_test(
        X, Y, 'm', 'medium_v1', 0, 'cuda', 'torch')

    run_forward_test(
        X, Y, 's', 'small_v1', 0, 'cuda', 'torch')

    run_forward_test(
        X, Y, 'l', 'large_v1', 15, 'cuda', 'torch')

    run_forward_test(
        X, Y, 'm', 'medium_v1', 15, 'cuda', 'torch')

    run_forward_test(
        X, Y, 's', 'small_v1', 15, 'cuda', 'torch')

    run_forward_test(
        X, Y, 'l', 'large_v1', -15, 'cpu', 'torch')

    run_forward_test(
        X, Y, 'm', 'medium_v1', -15, 'cpu', 'torch')

    run_forward_test(
        X, Y, 's', 'small_v1', -15, 'cpu', 'torch')

    run_forward_test(
        X, Y, 'l', 'large_v1', 0, 'cpu', 'torch')

    run_forward_test(
        X, Y, 'm', 'medium_v1', 0, 'cpu', 'torch')

    run_forward_test(
        X, Y, 's', 'small_v1', 0, 'cpu', 'torch')

    run_forward_test(
        X, Y, 'l', 'large_v1', 15, 'cpu', 'torch')

    run_forward_test(
        X, Y, 'm', 'medium_v1', 15, 'cpu', 'torch')

    run_forward_test(
        X, Y, 's', 'small_v1', 15, 'cpu', 'torch')
