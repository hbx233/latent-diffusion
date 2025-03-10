# Export diffusion models to onnx
import types
import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from optimization.ddim_trt import DDIMSamplerTRT

from onnx import shape_inference
import onnx_graphsurgeon as gs
import onnx
import onnxruntime as rt


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")  # TODO: check path

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.cpu()

def onnxruntime_check(onnx_path, input_dicts, torch_outputs):
    # onnx_model = onnx.load(onnx_path)
    # onnx.checker.check_model(onnx_model)
    sess = rt.InferenceSession(onnx_path)
    result = sess.run(None, input_dicts)
    print(result)

    for i in range(0, len(torch_outputs)):
        abs_diff = np.abs(result[i] - torch_outputs[i].detach().numpy())
        # Find the maximum difference
        print("Max parity error for {}th output: ".format(i), np.max(abs_diff))
        ret = np.allclose(result[i], torch_outputs[i].detach().numpy(), rtol=1e-03, atol=1e-05, equal_nan=False)
        if ret is False:
            print("Error onnxruntime_check")

def export_bert_model(
        opset_version,
        do_constant_folding,
        export_path,
        dynamic_batch_size,
        verbose=True,
    ):
    bert_model = model.cond_stage_model

    import types
    def forward(self, tokens):
        z = self.transformer(tokens, return_embeddings=True)
        return z
    
    bert_model.forward = types.MethodType(forward, bert_model)

    onnx_path = os.path.join(export_path, "BERT.onnx")

    tokens = torch.ones(1, 77, dtype=torch.int32)
    input_names = ["input_ids"]
    output_names = ["embedding"]
    dynamic_axes = {
        "input_ids": {0: "batch_size"},
        "embedding": {0: "batch_size"}
    } if dynamic_batch_size else None

    torch.onnx.export(
        bert_model,
        (tokens,),
        onnx_path,
        verbose=verbose,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

    output = bert_model(tokens)

    input_dicts = {"input_ids": tokens.numpy()}
    onnxruntime_check(onnx_path, input_dicts, [output])

def export_unet_model(
        opset_version,
        do_constant_folding,
        export_path,
        dynamic_batch_size, 
        H, 
        W, 
        verbose=True,
    ):
    unet = model.model.diffusion_model
    x = torch.randn(2, 4, H, W, dtype=torch.float32)
    timesteps = torch.randint(1, 10, (2,), dtype=torch.int32)
    context = torch.randn(2, 77, 1280, dtype=torch.float32)
    input_names = ["x", "timesteps", "context"]
    output_names = ["output"]
    dynamic_axes = {name:{0: "batch_size"} for name in (input_names + output_names)} if dynamic_batch_size else None

    onnx_file = onnx_file = os.path.join(export_path, "UNet.onnx")

    torch.onnx.export(
        unet,
        (x, timesteps, context),
        onnx_file,
        verbose = verbose,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        input_names=input_names,
        keep_initializers_as_inputs=True,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    outputs = [unet(x, timesteps, context)]
    input_dict = {
        "x": x.numpy(),
        "timesteps": timesteps.numpy(),
        "context": context.numpy(),
    }
    
    onnxruntime_check(onnx_file, input_dict, outputs)

def test_torch_unet_with_kv_caches():
    unet = model.model.diffusion_model
    x = torch.randn(2, 4, 32, 32, dtype=torch.float32)
    timesteps = torch.randint(1, 10, (2,), dtype=torch.int32)
    context = torch.randn(2, 77, 1280, dtype=torch.float32)
    # compute unet result with context kv caches
    context_kvcaches = unet.context_kvcaches(context)
    eps_with_kvcaches = unet.forward_with_context_kvcaches(x, timesteps, context_kvcaches)
    # compute unet with original model
    eps = unet(x, timesteps, context)
    print("UNet with kvcache error: ", torch.max(torch.abs(eps_with_kvcaches - eps)))

def export_unet_context_kv_caches_model(
        opset_version,
        do_constant_folding,
        export_path,
        dynamic_batch_size, 
        verbose=True,
):
    unet = model.model.diffusion_model
    context = torch.randn(2, 77, 1280, dtype=torch.float32)
    unet.forward = unet.context_kvcaches #bind to forward function as required by onnx export

    context_kvcaches_onnx_file = os.path.join(export_path, "unet_context_kvcaches_model.onnx")

    # Export context kv caches model
    input_names = ["context"]
    num_transformer_blocks = 16
    output_names = []
    for i in range(num_transformer_blocks):
        output_names += ["output_k_cache_{}".format(i), "output_v_caches_{}".format(i)]
    dynamic_axes = {name:{0: "batch_size"} for name in (input_names + output_names)} if dynamic_batch_size else None
    torch.onnx.export(
        unet,
        (context,),
        context_kvcaches_onnx_file,
        verbose = verbose,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        input_names=input_names,
        keep_initializers_as_inputs=True,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    context_kvcaches = unet.context_kvcaches(context)
    
    input_dict = {
        "context": context.numpy(),
    }
    onnxruntime_check(context_kvcaches_onnx_file, input_dict, context_kvcaches)


def export_unet_model_with_context_kvcaches(
        opset_version,
        do_constant_folding,
        export_path,
        dynamic_batch_size, 
        H, 
        W, 
        verbose=True,
):
    unet = model.model.diffusion_model
    x = torch.randn(2, 4, H, W, dtype=torch.float32)
    timesteps = torch.randint(1, 10, (2,), dtype=torch.int32)
    context = torch.randn(2, 77, 1280, dtype=torch.float32)
    unet.forward = types.MethodType(unet.context_kvcaches, unet) #bind to forward function as required by onnx export

    
    context_kvcaches_onnx_file = os.path.join(export_path, "unet_context_kvcaches_model.onnx")

    num_transformer_blocks = 16

    # Export unet model with kv caches as input
    unet_onnx_file = os.path.join(export_path, "unet_model_with_context_kvcaches.onnx")
    input_names = ["x", "timesteps"]
    for i in range(num_transformer_blocks):
        input_names += ["context_k_cache_{}".format(i), "context_v_cache_{}".format(i)]
    output_names = ["output"]
    dynamic_axes = {name:{0: "batch_size"} for name in (input_names + output_names)} if dynamic_batch_size else None
    unet.forward = unet.forward_with_context_kvcaches
    torch_context_kvcaches = unet.context_kvcaches(context)
    torch.onnx.export(
        unet,
        (x, timesteps, torch_context_kvcaches),
        unet_onnx_file,
        verbose = verbose,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        input_names=input_names,
        keep_initializers_as_inputs=True,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # fill input with onnx generated kv caches
    context_kvcache_input_dicts = {
        "context": context.numpy(),
    }
    sess = rt.InferenceSession(context_kvcaches_onnx_file)
    onnx_context_kvcaches = sess.run(None, context_kvcache_input_dicts)
    input_dict = {
        "x": x.numpy(),
        "timesteps": timesteps.numpy()
    }
    for i in range(num_transformer_blocks):
        input_dict["context_k_cache_{}".format(i)] = onnx_context_kvcaches[i*2]
        input_dict["context_v_cache_{}".format(i)] = onnx_context_kvcaches[i*2 + 1]

    # Compare torch end-to-end output and onnx end-to-end result
    outputs = [unet.forward_with_context_kvcaches(x, timesteps, torch_context_kvcaches)]
    onnxruntime_check(unet_onnx_file, input_dict, outputs)

def export_ddim_sampling_module(
        opset_version,
        do_constant_folding,
        export_path,
        dynamic_batch_size,
        H, 
        W,
        verbose=True,
):
    class DDIMSamplingModule(torch.nn.Module):
        def forward(self, x, e_t_uncond, e_t, a_t, a_prev, sigma_t, sqrt_one_minus_at):
            return DDIMSamplerTRT.sample_ddim_with_eps(x, e_t_uncond, e_t, a_t, a_prev, sigma_t, sqrt_one_minus_at)
    sampler = DDIMSamplingModule()
    x = torch.randn(1, 4, H, W, dtype=torch.float32)
    e_t_uncond = torch.randn(1, 4, H, W, dtype=torch.float32)
    e_t = torch.randn(1, 4, H, W, dtype=torch.float32)
    a_t = torch.tensor([[[[0.5]]]], dtype=torch.float32)
    a_prev = torch.tensor([[[[0.5]]]], dtype=torch.float32)
    sigma_t = torch.tensor([[[[0.5]]]], dtype=torch.float32)
    sqrt_one_minus_at = torch.tensor([[[[0.5]]]], dtype=torch.float32)

    ddim_sample_onnx_file = os.path.join(export_path, "ddim_sample.onnx")

    input_names = ['x', 'e_t_uncond', 'e_t', 'a_t', 'a_prev', 'sigma_t', 'sqrt_one_minus_at']
    output_names = ['x_prev_cat']

    torch.onnx.export(
        sampler,
        (x, e_t_uncond, e_t, a_t, a_prev, sigma_t, sqrt_one_minus_at),
        ddim_sample_onnx_file,
        verbose = verbose,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        input_names=input_names,
        output_names=output_names,
    )
    outputs = [sampler(x, e_t_uncond, e_t, a_t, a_prev, sigma_t, sqrt_one_minus_at)]

    input_dict = {
        'x': x.numpy(),
        'e_t_uncond': e_t_uncond.numpy(),
        'e_t': e_t.numpy(),
        'a_t': a_t.numpy(),
        'a_prev': a_prev.numpy(),
        'sigma_t': sigma_t.numpy(),
        'sqrt_one_minus_at': sqrt_one_minus_at.numpy(),
    }
    onnxruntime_check(ddim_sample_onnx_file, input_dict, outputs)

def export_first_stage_model(
        opset_version,
        do_constant_folding,
        export_path,
        dynamic_batch_size,
        verbose=True,
    ):
    # latent space to image space encoder decoder
    first_stage_model = model.first_stage_model
    first_stage_model.forward = first_stage_model.decode

    x = torch.randn(1, 4, 32, 32, dtype=torch.float32)
    input_names = ["x"]
    output_names = ["output"]
    dynamic_axes = {name:{0: "batch_size"} for name in (input_names + output_names)} if dynamic_batch_size else None

    onnx_file = os.path.join(export_path, "first_stage_model.onnx")
    torch.onnx.export(
        first_stage_model,
        (x,),
        onnx_file,
        verbose = verbose,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        input_names=input_names,
        keep_initializers_as_inputs=True,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    outputs = first_stage_model.decode(x)

    input_dict = {
        "x": x.numpy()
    }
    onnxruntime_check(onnx_file, input_dict, outputs)

if __name__ == "__main__":
    verbose = True
    opset_version = 18
    do_constant_folding = True
    dynamic_batch_size = False
    image_H = 256
    image_W = 256
    latent_downscale = 8
    assert(image_H % latent_downscale == 0)
    assert(image_W % latent_downscale == 0)
    latent_H = image_H // latent_downscale
    latent_W = image_W // latent_downscale
    export_path = "./onnx_{}".format(opset_version)

    os.makedirs(export_path, exist_ok=True)

    # export_bert_model(opset_version, do_constant_folding, export_path, dynamic_batch_size, verbose)
    
    # export_first_stage_model(opset_version, do_constant_folding, export_path, dynamic_batch_size, verbose)
    
    # UNet
    ## Original
    # export_unet_model(opset_version, do_constant_folding, export_path, dynamic_batch_size, latent_H, latent_W, verbose)

    ## With Explicit Context KV Caches
    # test_torch_unet_with_kv_caches()
    export_unet_context_kv_caches_model(opset_version, do_constant_folding, export_path, dynamic_batch_size, verbose)
    # export_unet_model_with_context_kvcaches(opset_version, do_constant_folding, export_path, dynamic_batch_size, latent_H, latent_W, verbose)

    ## DDIM sample
    # export_ddim_sampling_module(opset_version, do_constant_folding, export_path, dynamic_batch_size, latent_H, latent_W, verbose)