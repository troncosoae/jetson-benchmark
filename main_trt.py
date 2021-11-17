import numpy as np

from trt_lib.Interface import Interface
from importable_main import import_data
from data_worker.data_worker import split_into_batches
from torch_lib.data_worker import suit4torch


if __name__ == "__main__":

    target_dtype = np.float32

    X_data, Y_data = import_data()

    X, Y = suit4torch(X_data, Y_data)
    batch_size = 1
    batches = split_into_batches(X, Y, batch_size)
    X, Y = batches[2]

    X = X.float()
    Y = Y.float()

    X = np.array(X, dtype=target_dtype)
    Y = np.array(Y, dtype=target_dtype)

    path = "saved_nets/saved_onnx/torch_small_v1.trt"

    net_interface = Interface(
        path, X, batch_size=batch_size, n_classes=10,
        target_dtype=target_dtype)

    Y_pred = net_interface.predict_net(X)

    print(Y_pred)
