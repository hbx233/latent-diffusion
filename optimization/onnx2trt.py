import numpy as np
import os
import tensorrt as trt
from optimization.utils import *

from onnx import shape_inference
import onnx_graphsurgeon as gs
import onnx
import onnxruntime as rt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def tensorrt_check(onnx_path, onnx_input_dicts, plan_path, trt_input_dicts):
    # Load tensorrt engine
    trt_logger = trt.Logger(trt.Logger.VERBOSE)
    engine = load_engine(plan_path, trt_logger)
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
            assert trt_input_dicts[name].shape == shape
            assert trt_input_dicts[name].dtype == dtype
            bindings.append(trt_input_dicts[name].data_ptr())
        else:
            tensor = torch.empty(shape, dtype=dtype, device="cuda")
            output_tensors.append(tensor)
            bindings.append(tensor.data_ptr())
    
    context.execute_v2(bindings)

    # onnx_model = onnx.load(onnx_path)
    # onnx.checker.check_model(onnx_model)
    sess = rt.InferenceSession(onnx_path)
    result = sess.run(None, onnx_input_dicts)

    for i in range(0, len(output_tensors)):
        output_tensor = output_tensors[i].to(device='cpu')
        abs_diff = np.abs(result[i] - output_tensor.detach().numpy())
        print("=================")
        print(abs_diff)
        # Find the maximum difference
        print("TRT - ONNX max diff: ", np.max(abs_diff))
        ret = np.allclose(result[i], output_tensor.detach().numpy(), rtol=1e-03, atol=1e-05, equal_nan=False)
        if ret is False:
            print("Error onnxruntime_check")

def onnx2trt(onnx_file,
             plan_name,
            #  min_shapes,
            #  opt_shapes,
            #  max_shapes,
             max_workspace_size = 10<<30,
             fp16_mode=False,
             builder_opt_evel=5
    ):
    trt_logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

    parser = trt.OnnxParser(network, trt_logger)
    if not os.path.exists(onnx_file):
        raise RuntimeError("Failed finding onnx file!")
    print("Succeeded finding onnx file!")
    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read(), path=onnx_file):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX file!")
    print("Succeeded parsing ONNX file!")

    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)

    if builder_opt_evel:
        config.builder_optimization_level = builder_opt_evel

    profile = builder.create_optimization_profile()
    
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        if not input_tensor.is_shape_tensor:
            input_name = input_tensor.name
            input_shape = input_tensor.shape
            if -1 in input_shape:  # Only needed for dynamic input shapes
                min_shape = tuple(1 if dim == -1 else dim for dim in input_shape)   # Smallest batch size
                opt_shape = tuple(2 if dim == -1 else dim for dim in input_shape)   # Typical batch size
                max_shape = tuple(4 if dim == -1 else dim for dim in input_shape)   # Largest batch size
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                print(f"Optimization profile set for {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
            else:
                print(f"No dynamic axes: {input_shape}")
    config.add_optimization_profile(profile)
    engine = builder.build_serialized_network(network, config)
    if not engine:
        raise RuntimeError("build_serialized_network failed")
    print("Succeeded building engine!")

    with open(plan_name, "wb") as fout:
        fout.write(engine)

def export_bert_model(onnx_dir, engine_dir, fp16_mode):
    onnx_path = os.path.join(onnx_dir, "BERT.onnx")
    plan_path = os.path.join(engine_dir, "BERT.plan")

    onnx2trt(onnx_path, plan_path, fp16_mode=fp16_mode)

    tokens = torch.ones(1, 77, dtype=torch.int32)
    trt_input_dict = {
        "input_ids": tokens.to(device='cuda')
    }
    onnx_input_dict = {
        "input_ids": tokens.numpy()
    }
    tensorrt_check(onnx_path, onnx_input_dict, plan_path, trt_input_dict)
    print("======================= BERT onnx2trt done!")

def export_unet_model(onnx_dir, engine_dir, fp16_mode):
    onnx_path = os.path.join(onnx_dir, "UNet.onnx")
    plan_name = "UNet{}.plan".format("_fp16" if fp16_mode else "")
    plan_path = os.path.join(engine_dir, plan_name)

    # onnx2trt(onnx_path, plan_path, fp16_mode=fp16_mode)

    x = torch.randn(2, 4, 32, 32, dtype=torch.float32)
    timesteps = torch.randint(1, 10, (2,), dtype=torch.int32)
    context = torch.randn(2, 77, 1280, dtype=torch.float32)

    trt_input_dict = {
        "x": x.to(device="cuda"),
        "timesteps": timesteps.to(device="cuda"),
        "context": context.to(device="cuda"),
    }

    onnx_input_dict = {
        "x": x.numpy(),
        "timesteps": timesteps.numpy(),
        "context": context.numpy(),
    }
    tensorrt_check(onnx_path, onnx_input_dict, plan_path, trt_input_dict)
    print("======================= UNet onnx2trt done!")

def export_ddim_sampling_module(onnx_dir, engine_dir, fp16_mode):
    onnx_path = os.path.join(onnx_dir, "ddim_sample.onnx")
    plan_name = "ddim_sampler.plan"
    plan_path = os.path.join(engine_dir, plan_name)
    onnx2trt(onnx_path, plan_path)

    x = torch.randn(1, 4, 32, 32, dtype=torch.float32)
    e_t_uncond = torch.randn(1, 4, 32, 32, dtype=torch.float32)
    e_t = torch.randn(1, 4, 32, 32, dtype=torch.float32)
    a_t = torch.tensor([[[[0.5]]]], dtype=torch.float32)
    a_prev = torch.tensor([[[[0.5]]]], dtype=torch.float32)
    sigma_t = torch.tensor([[[[0.5]]]], dtype=torch.float32)
    sqrt_one_minus_at = torch.tensor([[[[0.5]]]], dtype=torch.float32)

    trt_input_dict = {
        'x': x.to(device="cuda"),
        'e_t_uncond': e_t_uncond.to(device="cuda"),
        'e_t': e_t.to(device="cuda"),
        'a_t': a_t.to(device="cuda"),
        'a_prev': a_prev.to(device="cuda"),
        'sigma_t': sigma_t.to(device="cuda"),
        'sqrt_one_minus_at': sqrt_one_minus_at.to(device="cuda"),
    }

    onnx_input_dict = {
        'x': x.numpy(),
        'e_t_uncond': e_t_uncond.numpy(),
        'e_t': e_t.numpy(),
        'a_t': a_t.numpy(),
        'a_prev': a_prev.numpy(),
        'sigma_t': sigma_t.numpy(),
        'sqrt_one_minus_at': sqrt_one_minus_at.numpy(),
    }
    tensorrt_check(onnx_path, onnx_input_dict, plan_path, trt_input_dict)

def export_unet_context_kvcaches_model(onnx_dir, engine_dir, fp16_mode):
    onnx_path = os.path.join(onnx_dir, "unet_context_kvcaches_model.onnx")
    plan_name = "unet_context_kvcaches_model{}.plan".format("_fp16" if fp16_mode else "")
    plan_path = os.path.join(engine_dir, plan_name)

    onnx2trt(onnx_path, plan_path, fp16_mode=fp16_mode)

    context = torch.randn(2, 77, 1280, dtype=torch.float32)

    trt_input_dict = {
        "context": context.to(device="cuda"),
    }

    onnx_input_dict = {
        "context": context.numpy(),
    }
    tensorrt_check(onnx_path, onnx_input_dict, plan_path, trt_input_dict)
    print("======================= UNet onnx2trt done!")

def export_unet_model_with_context_kvcaches(onnx_dir, engine_dir, fp16_mode):
    context_kvcaches_onnx_path = os.path.join(onnx_dir, "unet_context_kvcaches_model.onnx")
    unet_onnx_path = os.path.join(onnx_dir, "unet_model_with_context_kvcaches.onnx")
    context_kvcaches_plan_name = "unet_context_kvcaches_model{}.plan".format("_fp16" if fp16_mode else "")
    context_kvcaches_plan_path = os.path.join(engine_dir, context_kvcaches_plan_name)
    unet_plan_name = "unet_model_with_context_kvcaches{}.plan".format("_fp16" if fp16_mode else "")
    unet_plan_path = os.path.join(engine_dir, unet_plan_name)

    onnx2trt(unet_onnx_path, unet_plan_path, fp16_mode=fp16_mode)

    x = torch.randn(2, 4, 32, 32, dtype=torch.float32)
    timesteps = torch.randint(1, 10, (2,), dtype=torch.int32)
    context = torch.randn(2, 77, 1280, dtype=torch.float32)

    # Compute onnx context kv caches
    num_transformer_blocks = 16
    onnx_input_dict = {
        "context": context.numpy(),
    }
    sess = rt.InferenceSession(context_kvcaches_onnx_path)
    onnx_context_kvcaches = sess.run(None, onnx_input_dict)
    onnx_input_dict = {
        "x": x.numpy(),
        "timesteps": timesteps.numpy()
    }
    for i in range(num_transformer_blocks):
        onnx_input_dict["context_k_cache_{}".format(i)] = onnx_context_kvcaches[i*2]
        onnx_input_dict["context_v_cache_{}".format(i)] = onnx_context_kvcaches[i*2 + 1]

    # Compute trt context kv caches
    trt_input_dict = {
        "context": context.to(device="cuda"),
    }
    trt_logger = trt.Logger(trt.Logger.VERBOSE)
    engine = load_engine(context_kvcaches_plan_path, trt_logger)
    trt_context_kvcaches = trt_infer(engine, trt_input_dict)

    trt_input_dict = {
        "x": x.to(device="cuda"),
        "timesteps": timesteps.to(device="cuda"),
    }
    for i in range(num_transformer_blocks):
        trt_input_dict["context_k_cache_{}".format(i)] = trt_context_kvcaches[i*2]
        trt_input_dict["context_v_cache_{}".format(i)] = trt_context_kvcaches[i*2 + 1]
    tensorrt_check(unet_onnx_path, onnx_input_dict, unet_plan_path, trt_input_dict)
    print("======================= UNet onnx2trt done!")

def export_first_stage_model(onnx_dir, engine_dir, fp16_mode):
    onnx_path = os.path.join(onnx_dir, "first_stage_model.onnx")
    plan_path = os.path.join(engine_dir, "first_stage_model.plan")

    onnx2trt(onnx_path, plan_path, fp16_mode=fp16_mode)

    x = torch.randn(1, 4, 32, 32, dtype=torch.float32)
    trt_input_dict = {
        "x": x.to(device="cuda")
    }

    onnx_input_dict = {
        "x": x.numpy()
    }
    tensorrt_check(onnx_path, onnx_input_dict, plan_path, trt_input_dict)
    print("======================= Decoder  onnx2trt done!")

def main():
    onnx_dir = "./onnx_18"
    plan_dir = "./engine"
    fp16_mode = True
    os.makedirs(plan_dir, exist_ok=True)
    # export_bert_model(onnx_dir, plan_dir, fp16_mode=fp16_mode)
    # export_first_stage_model(onnx_dir, plan_dir, fp16_mode=fp16_mode) # Error: 0.0014116913

    # UNet
    ## Original UNet
    # export_unet_model(onnx_dir, plan_dir, fp16_mode) # TRT - ONNX max diff:  0.00064766407

    ## UNet with Context KV Caches
    # export_unet_context_kvcaches_model(onnx_dir, plan_dir, fp16_mode) # TRT - ONNX max diff:  0.00064766407
    export_unet_model_with_context_kvcaches(onnx_dir, plan_dir, fp16_mode)

    # DDIM Sampler
    # export_ddim_sampling_module(onnx_dir, plan_dir, fp16_mode)

if __name__ == '__main__':
    main()