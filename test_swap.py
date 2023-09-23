import os
import numpy as np
import cv2
import torch
from basicsr.utils import tensor2img
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.inference_base import (diffusion_inference, get_adapters, get_base_argument_parser, get_sd_models, str2bool)
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import (ExtraCondition, get_adapter_feature, get_cond_model)

torch.set_grad_enabled(False)

# Use two of diffusion line and swap the part of tensor in latent space.



def main():
    supported_cond = [e.name for e in ExtraCondition]
    parser = get_base_argument_parser()
    parser.add_argument(
        '--newprompt',
        # TODO: As a LLM prompt for Edition?
        type=str,
        default="nothing",
        help='another prompt in double line mode'
    )
    parser.add_argument(
        '--which_cond',
        type=str,
        default="None",
        choices=supported_cond,
        help='which condition modality you want to test',
    )
    parser.add_argument(
        '--keep',
        type=str2bool,
        default=True,
        help='whether to keep swapping tensors in latent space',
    )
    parser.add_argument(
        '--swapW',
        type=int,
        default=128,
        help='less than W'
    )
    parser.add_argument(
        '--swapH',
        type=int,
        default=128,
        help='less than H'
    )
    # swap or just use a crop area in one latent image to replace the corresponding area in another ?
    # TODO: (do the swap or the replacement ?)
    parser.add_argument(
        '--endStep',
        type=int,
        default=25,
        help='when to end up swapping tensors in latent space'
    )
    parser.add_argument(
        '--is_double',
        type=str2bool,
        default=True,
        help='use two stable-diffusion base line two enable the swap method'
    )

    
    
    opt = parser.parse_args()
    which_cond = opt.which_cond
    if opt.outdir is None:
        opt.outdir = f'outputs/test-{which_cond}/' + opt.adapter_ckpt.split('/')[-1].strip('.pth')
    os.makedirs(opt.outdir, exist_ok=True)
    if opt.resize_short_edge is None:
        print(f"you don't specify the resize_shot_edge, so the maximum resolution is set to {opt.max_resolution}")
    opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # support two test mode: single image test, and batch test (through a txt file)
    if opt.prompt.endswith('.txt'):
        assert opt.prompt.endswith('.txt')
        image_paths = []
        prompts = []
        with open(opt.prompt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                image_paths.append(line.split('; ')[0])
                prompts.append(line.split('; ')[1])
    else:
        image_paths = [opt.cond_path] if opt.cond_path != None else [None]
        prompts = [opt.prompt]
    print(image_paths)

    # prepare models
    sd_model, sampler = get_sd_models(opt)
    if opt.which_cond != "None":
        adapter = get_adapters(opt, getattr(ExtraCondition, which_cond))
    cond_model = None
    if opt.cond_inp_type == 'image' and opt.which_cond != "None":
        cond_model = get_cond_model(opt, getattr(ExtraCondition, which_cond))
    if opt.which_cond != "None":
        process_cond_module = getattr(api, f'get_cond_{which_cond}')

    # inference
    with torch.inference_mode(), \
            sd_model.ema_scope(), \
            autocast('cuda'):
        for test_idx, (cond_path, prompt) in enumerate(zip(image_paths, prompts)):
            seed_everything(opt.seed)
            for v_idx in range(opt.n_samples):
                # seed_everything(opt.seed+v_idx+test_idx)
                cond = process_cond_module(opt, cond_path, opt.cond_inp_type, cond_model) if cond_path != None else None

                base_count = len(os.listdir(opt.outdir)) // 2
                if cond != None:
                    cv2.imwrite(os.path.join(opt.outdir, f'{base_count:05}_{which_cond}.png'), tensor2img(cond))

                if opt.which_cond != "None" and cond != None:
                    adapter_features, append_to_context = get_adapter_feature(cond, adapter)
                else:
                    adapter_features, append_to_context = None, None
                opt.prompt = prompt

                result = diffusion_inference(opt, sd_model, sampler, adapter_features, append_to_context)
                if opt.is_double:
                    assert isinstance(result, list), '?'
                    assert len(result) == 2, '??'
                a, b = tensor2img(result[0]), tensor2img(result[1])
                # print(type(result))
                # print(result)
                # result = np.ndarray(result, dtype=np.float32)
                cv2.imwrite(os.path.join(opt.outdir, f'{base_count:05}_result_a.png'), a)
                cv2.imwrite(os.path.join(opt.outdir, f'{base_count:05}_result_b.png'), b)
                


if __name__ == '__main__':
    main()
