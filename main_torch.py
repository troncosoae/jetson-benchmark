import time
import torch
import sys
import os
import csv

from jtop_lib.JtopAdapter import JtopAdapter
from data_worker.data_worker import split_into_batches
from importable_main import import_data, write_csv
from torch_lib.data_worker import suit4torch
from torch_lib.Interface import Interface as torchInterface
from torch_lib.Nets import LargeNet as torchLNet, \
    MediumNet as torchMNet, SmallNet as torchSNet


def execute_net_torch(
        net_interface, X_data, Y_data, batch_size, priority, loops=1,
        device='cuda', echo=True):

    if sys.platform == 'linux':
        os.nice(priority)

    if device not in ['cuda', 'cpu']:
        raise Exception("'device' parameter must be one of 'cuda' or 'cpu'")

    if device == 'cuda':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if echo and not torch.cuda.is_available():
            print("'cuda' device is not available. Using 'cpu' instead.")
    else:
        device = torch.device('cpu')

    net_interface.to_device(device)

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
                X_batch = X_batch.to(device)
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


def run_forward_test_torch(
        X_data, Y_data, net_size, saved_net_path, priority, device):

    framework = 'torch'

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
    tracked_values, cols1, batch_exec_times, cols2 = execute_net_torch(
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

    try:
        sizes = [sys.argv[1]]
        priorities = [int(sys.argv[2])]
        devices = [sys.argv[3]]
    except IndexError:
        sizes = ['large_v1', 'medium_v1', 'small_v1']
        priorities = [-15, 0, 15]
        devices = ['cuda', 'cpu']

    for size in sizes:
        for priority in priorities:
            for device in devices:
                print(size, priority, device)
                run_forward_test_torch(
                    X, Y, size[0], size, priority, device)
