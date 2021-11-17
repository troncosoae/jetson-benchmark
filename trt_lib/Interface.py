import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


from importable_main import import_data
from data_worker.data_worker import split_into_batches
from torch_lib.data_worker import suit4torch


class Interface:
    def __init__(self, path, dummy_input, batch_size, n_classes, **kwargs):
        self.path = path
        self.target_dtype = kwargs.get('target_dtype', np.float32)

        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(self.path, "rb") as file:
            self.engine = self.runtime.deserialize_cuda_engine(file.read())
        self.context = self.engine.create_execution_context()

        self.output = np.empty(
            [batch_size, n_classes], dtype=self.target_dtype)
        self.d_input = cuda.mem_alloc(1 * dummy_input.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)

        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()

    def predict_net(self, X):
        cuda.memcpy_htod_async(self.d_input, X, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        self.stream.synchronize()

        return self.output


if __name__ == "__main__":
    pass
