import torch
import numpy as np
import tensorrt as trt

def load_engine(plan_file_path, logger):
    with open(plan_file_path, "rb") as f:
        engine_data = f.read()
    # Create a runtime object
    runtime = trt.Runtime(logger)
    # Deserialize engine
    engine = runtime.deserialize_cuda_engine(engine_data)
    if engine:
        print("Successfully loaded TensorRT engine from plan file {}.".format(plan_file_path))
    else:
        print("Failed to load TensorRT engine from plan file {}.".format(plan_file_path))
    return engine

def trt_get_torch_shape(engine, name):
    return tuple(engine.get_tensor_shape(name))

def trt_get_torch_dtype(engine, name):
    dtype_np = trt.nptype(engine.get_tensor_dtype(name))
    return torch.from_numpy(np.array([], dtype=dtype_np)).dtype

def trt_get_engine_io_info(engine, name):
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = trt_get_torch_shape(engine, name)
        dtype = trt_get_torch_dtype(engine, name)
        print("Name: {}, mode: {}, shape: {}, dtype: {}".format(name, mode, shape, dtype))

def trt_get_tensor_bindings(engine, tensors):
    bindings = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = trt_get_torch_shape(engine, name)
        dtype = trt_get_torch_dtype(engine, name)
        # Check shape and dtype
        assert shape == tensors[i].shape, "Tensor shape mismatch, name: {}, received: {}, trt: {}".format(name, tensors[i].shape, shape)
        assert dtype == tensors[i].dtype, "Tensor dtype mismatch, name: {}, received: {}, trt: {}".format(name, tensors[i].dtype, dtype)
        bindings.append(tensors[i].data_ptr())
    return bindings

def trt_set_tensor_address(context, bindings):
    for i in range(context.engine.num_io_tensors):
        name = context.engine.get_tensor_name(i)
        context.set_tensor_address(name, bindings[i])

def trt_infer(engine, input_dict):
    context = engine.create_execution_context()
    bindings = []
    output_tensors = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = trt_get_torch_shape(engine, name)
        dtype = trt_get_torch_dtype(engine, name)
        print(name, shape, dtype)
        if mode == trt.TensorIOMode.INPUT:
            assert input_dict[name].shape == shape
            assert input_dict[name].dtype == dtype
            bindings.append(input_dict[name].data_ptr())
        else:
            tensor = torch.empty(shape, dtype=dtype, device="cuda")
            output_tensors.append(tensor)
            bindings.append(tensor.data_ptr())
    
    context.execute_v2(bindings)
    return output_tensors