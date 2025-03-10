import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from optimization.ddim_trt import DDIMSamplerTRT

import tensorrt as trt
from optimization.utils import *

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    opt = parser.parse_args()

    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")  # TODO: check path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
        # sampler = DDIMSamplerTRT(model,ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt


    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    logger = trt.Logger(trt.Logger.VERBOSE)
    bert_engine = load_engine("./engine/BERT.plan", logger)
    bert_context = bert_engine.create_execution_context()
    c = torch.empty((1,77,1280), dtype=torch.float32, device='cuda')
    uc = torch.empty((1,77,1280), dtype=torch.float32, device='cuda')

    decode_engine = load_engine("./engine/first_stage_model.plan", logger)
    decode_context = decode_engine.create_execution_context()

    num_iter = 6
    warmup_iters = 3

    use_trt = True

    all_samples=list()
    with torch.autograd.profiler.profile(enabled=False):
        with torch.no_grad():
            with model.ema_scope():
                for n in trange(num_iter, desc="Sampling"):
                    if n == warmup_iters:
                        torch.cuda.cudart().cudaProfilerStart()
                    torch.cuda.nvtx.range_push("iteration{}".format(n))
                    torch.cuda.nvtx.range_push("BERT")
                    if use_trt:
                        uc_tokens = model.cond_stage_model.tknz_fn([""]).to(dtype=torch.int32, device='cuda')
                        c_tokens = model.cond_stage_model.tknz_fn([prompt]).to(dtype=torch.int32, device='cuda')
                        bindings = trt_get_tensor_bindings(bert_engine, [uc_tokens, uc])
                        bert_context.execute_v2(bindings)
                        bindings = trt_get_tensor_bindings(bert_engine, [c_tokens, c])
                        bert_context.execute_v2(bindings)
                    else:
                        uc_torch = model.get_learned_conditioning([""])
                        c_torch = model.get_learned_conditioning([prompt])
                    torch.cuda.nvtx.range_pop()
                    shape = [4, opt.H//8, opt.W//8]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=c,
                                                     batch_size=opt.n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta)
                    torch.cuda.nvtx.range_push("Decoder")
                    if use_trt:
                        # image_tensor = model.decode_first_stage(samples_ddim)
                        samples_ddim = 1. / model.scale_factor * samples_ddim
                        image_tensor= torch.empty((1,3,256,256), dtype=torch.float32, device='cuda')
                        decode_bindings = [samples_ddim.data_ptr(), image_tensor.data_ptr()]
                        decode_context.execute_v2(decode_bindings)
                    else:
                        image_tensor = model.decode_first_stage(samples_ddim)
                    torch.cuda.nvtx.range_pop()
                    image_tensor = torch.clamp((image_tensor+1.0)/2.0, min=0.0, max=1.0)

                    for x_sample in image_tensor:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                        base_count += 1
                    all_samples.append(image_tensor)
                    torch.cuda.nvtx.range_pop()
                torch.cuda.cudart().cudaProfilerStop()

    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=opt.n_samples)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.png'))

    print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")