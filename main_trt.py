import numpy as np
import argparse

from trt_lib.Interface import Interface
from importable_main import import_data
from data_worker.data_worker import split_into_batches
from torch_lib.data_worker import suit4torch


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
    args = parser.parse_args()
    return args


def export_onnx_main():
    pass


def run_trt_main():
    pass


if __name__ == "__main__":

    args = get_args()

    if args.export_onnx:
        print('export onnx')
    else:
        print('normal')

    target_dtype = np.float32

    X_data, Y_data = import_data()

    X, Y = suit4torch(X_data, Y_data)
    batch_size = 10
    batches = split_into_batches(X, Y, batch_size)
    dummy_batch, _ = batches[0]

    X = X.float()
    Y = Y.float()

    X = np.array(X, dtype=target_dtype)
    Y = np.array(Y, dtype=target_dtype)
    X = np.ascontiguousarray(X)

    path = "saved_nets/saved_onnx/torch_small_v1.trt"

    net_interface = Interface(
        path, X, batch_size=batch_size, n_classes=10,
        target_dtype=target_dtype)

    Y_pred = net_interface.predict_net(X)

    print(Y_pred)
