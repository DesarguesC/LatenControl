"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
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
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                    1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

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
               features_adapter=None,
               append_to_context=None,
               cond_tau=0.4,
               style_cond_tau=1.0,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):

        is_double = kwargs['is_double'] if 'is_double' in kwargs.keys else None
        swap_shape = kwargs['swap_shape'] if 'swap_shape' in kwargs.keys and is_double is not None else None
        endStep = kwargs['endStep'] if 'endStep' in kwargs.keys and is_double is not None else None
        if is_double is not None and is_double:
            assert conditioning[1] is not None, 'conditioning contains None while doubling line'
        elif is_double is None:
            assert conditioning[0] is None, 'conditioning contains tow valid elements while not doubling'

        if conditioning is not None:
            if isinstance(conditioning[0], dict):
                cbs = conditioning[list(conditioning[0].keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            elif isinstance(conditioning[1], dict):
                cbs = conditioning[list(conditioning[1].keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            elif not is_double:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
            elif is_double:
                if conditioning[0].shape[0] != batch_size:
                    print(f"Warning: Got {conditioning[0].shape[0]} conditionings but batch-size is {batch_size}")
                if conditioning[1].shape[0] != batch_size:
                    print(f"Warning: Got {conditioning[1].shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        endStep = (C, endStep[0], endStep[1]) if endStep is not None else None

        print(f'Data shape for DDIM sampling is {size}, eta {eta}')


        """
        
        ddim_sampling return
        
            imgs, pred_x0s = outs
            # img, img_ = imgs[0], imgs[1]
            pred_x0, pred_x0_ = pred_x0s[0], pred_x0s[0]

            if callback: callback(i)
            if img_callback:
                img_callback(pred_x0, i)
                img_callback(pred_x0_, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(imgs)
                intermediates['pred_x0'].append(pred_x0s)

            return imgs, intermediates
        
        """



        samples, intermediates = self.ddim_sampling(conditioning, size,
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
                                                    features_adapter=features_adapter,
                                                    append_to_context=append_to_context,
                                                    cond_tau=cond_tau,
                                                    style_cond_tau=style_cond_tau,
                                                    is_double=is_double,
                                                    swap_shape=swap_shape,
                                                    endStep=endStep
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, features_adapter=None,
                      append_to_context=None, cond_tau=0.4, style_cond_tau=1.0, **kwargs):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
            img_ = torch.randn(shape, device=device)
        else:
            img = x_T
            img_ = x_T

        is_double = kwargs['is_double'] if 'is_double' in kwargs.keys else None
        swap_shape = kwargs['swap_shape'] if 'swap_shape' in kwargs.keys and is_double is not None else None
        endStep = kwargs['endStep'] if 'endStep' in kwargs.keys and is_double is not None else None

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        intermediates_ = {'x_inter': [img_], 'pred_x0': [img_]}

        _intermediates_ = [intermediates, intermediates_]

        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None

                print(x0.shape)  # why x0 is not None ???

                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
                img_ = img_orig * mask + (1. - mask) * img_

                imgs = [img, img_]

                # outs = [x_prev, x_prev_], [pred_x0, pred_x0_]

                outs = self.p_sample_ddim(imgs, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                          quantize_denoised=quantize_denoised, temperature=temperature,
                                          noise_dropout=noise_dropout, score_corrector=score_corrector,
                                          corrector_kwargs=corrector_kwargs,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning,
                                          features_adapter=None if index < int(
                                              (1 - cond_tau) * total_steps) else features_adapter,
                                          append_to_context=None if index < int(
                                              (1 - style_cond_tau) * total_steps) else append_to_context,
                                          is_double=is_double, swap_shape=swap_shape, endStep=endStep   # swap method
                                          )



            imgs, pred_x0s = outs
            # img, img_ = imgs[0], imgs[1]
            pred_x0, pred_x0_ = pred_x0s[0], pred_x0s[0]

            if callback: callback(i)
            if img_callback:
                img_callback(pred_x0, i)
                img_callback(pred_x0_, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(imgs)
                intermediates['pred_x0'].append(pred_x0s)

        return imgs, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, features_adapter=None,
                      append_to_context=None, **kwargs):
        b, *_, device = *x.shape, x.device
        # x: imgs

        is_double = kwargs['is_double'] if 'is_double' in kwargs.keys else None
        swap_shape = kwargs['swap_shape'] if 'swap_shape' in kwargs.keys and is_double is not None else None
        endStep = kwargs['endStep'] if 'endStep' in kwargs.keys and is_double is not None else None



        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            if append_to_context is not None:
                model_output = self.model.apply_model(x[0], t, torch.cat([c[0], append_to_context], dim=1),
                                                      features_adapter=features_adapter)
                model_output_ = self.model.apply_model(x[1], t, torch.cat([c[1], append_to_context], dim=1),
                                                      features_adapter=features_adapter) if c[1] is not None else None
            else:
                model_output = self.model.apply_model(x[0], t, c[0], features_adapter=features_adapter)
                model_output_ = self.model.apply_model(x[1], t, c[1], features_adapter=features_adapter) if c[1] is not None else None
        else:
            print('Double Line Not Yet Implemented')
            assert False, 'Double Line Not Implement Error'
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([
                            unconditional_conditioning[k][i],
                            c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([
                            unconditional_conditioning[k],
                            c[k]])
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            else:
                if append_to_context is not None:
                    pad_len = append_to_context.size(1)
                    new_unconditional_conditioning = torch.cat(
                        [unconditional_conditioning, unconditional_conditioning[:, -pad_len:, :]], dim=1)
                    new_c = torch.cat([c, append_to_context], dim=1)
                    c_in = torch.cat([new_unconditional_conditioning, new_c])
                else:
                    c_in = torch.cat([unconditional_conditioning, c])
            model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in, features_adapter=features_adapter).chunk(2)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            print('IN v')
            e_t = self.model.predict_eps_from_z_and_v(x[0], t, model_output)   # the noise in next turn
            e_t_ = self.model.predict_eps_from_z_and_v(x[1], t, model_output_) if is_double else None
        else:
            print('NOT IN v')
            e_t = model_output
            e_t_ = model_output_ if is_double else None

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x[0], t, c, **corrector_kwargs)
            e_t_ = score_corrector.modify_score(self.model, e_t_, x[1], t, c_, **corrector_kwargs) if is_double else None

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x[0] - sqrt_one_minus_at * e_t) / a_t.sqrt()
            pred_x0_ = (x[1] - sqrt_one_minus_at * e_t_) / a_t.sqrt() if is_doouble else None
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x[0], t, model_output)
            pred_x0_ = self.model.predict_start_from_z_and_v(x[1], t, model_output_) if is_doouble else None

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            pred_x0_, _, *_ = self.model.first_stage_model.quantize(pred_x0_)

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x[0].shape, device, repeat_noise) * temperature

        dir_xt_ = (1. - a_prev - sigma_t ** 2).sqrt() * e_t_
        noise_ = sigma_t * noise_like(x[1].shape, device, repeat_noise) * temperature

        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            noise_ = torch.nn.functional.dropout(noise_, p=noise_dropout)

        if t < endStep:
            pred_x0, pred_x0_ = SWAP_latent_img([pred_x0, pred_x0_], swap_shape)
        # TODO: swap tensors between two lines

        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        x_prev_ = a_prev.sqrt() * pred_x0_ + dir_xt_ + noise_



        return [x_prev, x_prev_], [pred_x0, pred_x0_]

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec

def SWAP_latent_img(x, swap_shape, start=1/4, throughout_channel=True):
    # start form 1/4 length of the width
    assert len(swap_shape)==2, 'shape exception'
    H, W = swap_shape
    # swap_shape: H, W
    if isinstance(x, list):
        assert len(x)==2, 'list length exception'
        assert x[1] is not None, 'element exception'
        a, b = x[0], x[1]
        start_point_H, start_point_W = (int)(H*start), (int)(W*start)
        end_point_H, end_point_W = start_point_H + H + 1, start_point_W + W + 1
        if throughout_channel:
            temp = a[:, start_point_H:end_point_H, start_point_W:end_point_W]
            a[:, start_point_H:end_point_H, start_point_W:end_point_W] \
                = b[:, start_point_H:end_point_H, start_point_W:end_point_W]
            b[:, start_point_H:end_point_H, start_point_W:end_point_W] = temp
        else:
            temp = a[0][start_point_H:end_point_H, start_point_W:end_point_W]
            a[0][start_point_H:end_point_H, start_point_W:end_point_W] \
                = b[0][start_point_H:end_point_H, start_point_W:end_point_W]
            b[0][start_point_H:end_point_H, start_point_W:end_point_W] = temp
    else:
        raise RuntimeError('Unable to swap for lack of swap target')

        return [a,b]
