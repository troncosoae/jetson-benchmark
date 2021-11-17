import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy


from importable_main import import_data
from data_worker.data_worker import split_into_batches
from torch_lib.data_worker import suit4torch


if __name__ == "__main__":
    pass
