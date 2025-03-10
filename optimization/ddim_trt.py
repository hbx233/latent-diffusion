"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
import tensorrt as trt
from optimization.utils import *

from cuda import cudart
def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
         raise RuntimeError(f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t")
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None

class DDIMSamplerTRT(object):
    def __init__(self, model, ddim_num_steps, ddim_eta, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

        self.fp16_mode = True
        self.use_kv_cache = True
        self.use_cuda_graph = True

        logger = trt.Logger(trt.Logger.VERBOSE)
        sampler_engine_path = "./engine/ddim_sampler.plan"
        self.sampler_engine = load_engine(sampler_engine_path, logger)
        self.sampler_context = self.sampler_engine.create_execution_context()

        self.stream = CUASSERT(cudart.cudaStreamCreate())

        if self.use_kv_cache:
            # For UNet with context KV caches
            unet_context_kvcaches_engine_path = "./engine/unet_context_kvcaches_model{}.plan".format("_fp16" if self.fp16_mode else "")
            unet_with_cache_engine_path = "./engine/unet_model_with_context_kvcaches{}.plan".format("_fp16" if self.fp16_mode else "")
            self.unet_context_kvcaches_engine = load_engine(unet_context_kvcaches_engine_path, logger)
            self.unet_context_kvcaches_context = self.unet_context_kvcaches_engine.create_execution_context()
            self.unet_with_cache_engine = load_engine(unet_with_cache_engine_path, logger)
            self.unet_with_cache_context = self.unet_with_cache_engine.create_execution_context()
            self.unet_with_cache_cuda_graph_instance = None
            ## Allocate buffers for kv caches to be maintained in the entire diffusion pass
            self.unet_context_kvcache_tensors = []
            self.unet_context_kvcache_ptrs = []
            for i in range(self.unet_context_kvcaches_engine.num_io_tensors):
                name = self.unet_context_kvcaches_engine.get_tensor_name(i)
                mode = self.unet_context_kvcaches_engine.get_tensor_mode(name)
                shape = trt_get_torch_shape(self.unet_context_kvcaches_engine, name)
                dtype = trt_get_torch_dtype(self.unet_context_kvcaches_engine, name)
                if mode == trt.TensorIOMode.OUTPUT:
                    tensor = torch.empty(shape, dtype=dtype, device='cuda')
                    self.unet_context_kvcache_tensors.append(tensor)
                    self.unet_context_kvcache_ptrs.append(tensor.data_ptr())
        else:
            unet_engine_path = "./engine/UNet{}.plan".format("_fp16" if self.fp16_mode else "")
            self.unet_engine = load_engine(unet_engine_path, logger)
            self.unet_context = self.unet_engine.create_execution_context()
            for i in range(self.unet_engine.num_io_tensors):
                name = self.unet_engine.get_tensor_name(i)
                mode = self.unet_engine.get_tensor_mode(name)
                shape = self.unet_engine.get_tensor_shape(name)
                print(name, mode, shape)
        # Need reset for each sample generation
        self.x_T = torch.empty((2, 4, 32, 32), dtype=torch.float32, device='cuda')
        self.x_T_ptr = self.x_T.data_ptr()
        # Will be overwritten, no need to reset
        self.eps = torch.empty((2, 4, 32, 32), dtype=torch.float32, device='cuda')
        self.eps_ptr = self.eps.data_ptr()

         # Make schedule for DDIM generation
        self.make_schedule(ddim_num_steps=ddim_num_steps, batch_size=1, device=torch.device('cuda'), ddim_eta=ddim_eta)

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, batch_size, device, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

        # Pre-allocate tensors to save fill 
        # select parameters corresponding to the currently considered timestep
        self.a_t_tensors = [torch.full((batch_size, 1, 1, 1), ddim_alpha, device=device) for ddim_alpha in self.ddim_alphas]
        self.a_prev_tensors = [torch.full((batch_size, 1, 1, 1), ddim_alpha_prev, device=device) for ddim_alpha_prev in self.ddim_alphas_prev]
        self.sigma_t_tensors = [torch.full((batch_size, 1, 1, 1), ddim_sigma, device=device) for ddim_sigma in self.ddim_sigmas]
        self.sqrt_one_minus_at_tensors = [torch.full((batch_size, 1, 1, 1), ddim_sqrt_one_minus_alpha,device=device) for ddim_sqrt_one_minus_alpha in self.ddim_sqrt_one_minus_alphas]

        e_t_uncond, e_t = self.eps.chunk(2)
        self.sampler_bindings = [[self.x_T.data_ptr(), e_t_uncond.data_ptr(), e_t.data_ptr(), self.a_t_tensors[index].data_ptr(), self.a_prev_tensors[index].data_ptr(), self.sigma_t_tensors[index].data_ptr(), self.sqrt_one_minus_at_tensors[index].data_ptr(), self.x_T.data_ptr()] for index in range(len(self.ddim_alphas))]
        
        self.time_range = np.flip(self.ddim_timesteps)
        self.t_tensors = [torch.cat([torch.full((1,), step, device=device, dtype=torch.int32)] * 2) for step in self.time_range]
        self.t_ptrs = [t_tensor.data_ptr() for t_tensor in self.t_tensors]

        # Fixed t_ptr for cuda graph
        self.t_tensor = torch.empty((2,), device='cuda', dtype=torch.int32)
        self.t_ptr = self.t_tensor.data_ptr()
        @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        # self.make_schedule(ddim_num_steps=S, batch_size=batch_size, device=self.model.betas.device, ddim_eta=eta, verbose=verbose)

        samples, intermediates = self.ddim_sampling_trt(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates
    
    def reset_image_buffers(self):
        # Reset the concatinated image buffer to random noise
        self.x_T.data.copy_(torch.cat([torch.randn(1,4,32,32, dtype=torch.float32, device='cuda')]*2))
    
    @torch.no_grad()
    def ddim_sampling_trt(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        assert b == 1, "Currently only support static shape"
        self.reset_image_buffers()

        total_steps = self.ddim_timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(self.time_range, desc='DDIM Sampler', total=total_steps)

        c_in = torch.cat([unconditional_conditioning, cond])
        c_in_ptr = c_in.data_ptr()
        if self.use_kv_cache:
            torch.cuda.nvtx.range_push("Context KV Cache")
            self.update_context_kv_caches_trt(c_in)
            torch.cuda.nvtx.range_pop()

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            self.p_sample_ddim_trt(self.x_T_ptr, c_in_ptr, i, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            # Swap x_T and x_prev
            # self.sampler_binding_idx = (self.sampler_binding_idx + 1) % 2
        return self.x_T[0:1], None
    
    def p_sample_ddim_trt(self, x_T_ptr, c_in_ptr, t_idx, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        torch.cuda.nvtx.range_push("Noise Pred")
        if self.use_kv_cache:
            # diffusion_bindings = trt_get_tensor_bindings(self.unet_with_cache_engine, [x_in, t_in] + self.unet_context_kvcache_ptrs + [self.eps]) # diffusion input
            #unet_bindings = [x_T_ptr, self.t_ptrs[t_idx]] + self.unet_context_kvcache_ptrs + [self.eps_ptr]
            #trt_set_tensor_address(self.unet_with_cache_context, unet_bindings)
            if self.use_cuda_graph:
                # Set context tensor address
                # Need to copy t to the fixed address since trt io address cannot be changed once cuda graph is captured
                self.t_tensor.data.copy_(self.t_tensors[t_idx])
                if self.unet_with_cache_cuda_graph_instance is not None:
                    CUASSERT(cudart.cudaGraphLaunch(self.unet_with_cache_cuda_graph_instance, self.stream))
                    CUASSERT(cudart.cudaStreamSynchronize(self.stream))
                else:
                    trt_set_tensor_address(self.unet_with_cache_context, [x_T_ptr, self.t_ptr] + self.unet_context_kvcache_ptrs + [self.eps_ptr])
                    # do inference before CUDA graph capture
                    noerror = self.unet_with_cache_context.execute_async_v3(self.stream)
                    if not noerror:
                        raise ValueError(f"ERROR: inference failed.")
                    # capture cuda graph
                    CUASSERT(cudart.cudaStreamBeginCapture(self.stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal))
                    self.unet_with_cache_context.execute_async_v3(self.stream)
                    self.graph = CUASSERT(cudart.cudaStreamEndCapture(self.stream))
                    self.unet_with_cache_cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiate(self.graph, 0))
            else:
                unet_bindings = [x_T_ptr, self.t_ptrs[t_idx]] + self.unet_context_kvcache_ptrs + [self.eps_ptr]
                trt_set_tensor_address(self.unet_with_cache_context, unet_bindings)
                # success = self.unet_with_cache_context.execute_v2(unet_bindings)
                noerror = self.unet_with_cache_context.execute_async_v3(self.stream)
                cudart.cudaStreamSynchronize(self.stream)
                if not noerror:
                    raise ValueError(f"ERROR: inference failed.")
        else:
            # diffusion_bindings = trt_get_tensor_bindings(self.unet_engine, [x_T_ptr, t_ptr, c_in, self.eps]) # diffusion input
            success = self.unet_context.execute_v2([x_T_ptr, self.t_ptrs[t_idx], c_in_ptr, self.eps_ptr])
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("Sample")
        trt_set_tensor_address(self.sampler_context, self.sampler_bindings[index])
        self.sampler_context.execute_async_v3(self.stream)
        CUASSERT(cudart.cudaStreamSynchronize(self.stream))
        torch.cuda.nvtx.range_pop()
    
    def sample_ddim_with_eps(x, e_t_uncond, e_t, a_t, a_prev, sigma_t, sqrt_one_minus_at):
        e_t = e_t_uncond + 10.0 * (e_t - e_t_uncond)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * torch.randn(x.shape, device=x.device, dtype=x.dtype)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        output = torch.cat([x_prev] * 2)
        return output
    
    def update_context_kv_caches_trt(self, context):
        print("Update UNet Context KV Caches")
        success = self.unet_context_kvcaches_context.execute_v2([context.data_ptr()] + self.unet_context_kvcache_ptrs)
        assert success, "TRT Context KV Cache inference error"