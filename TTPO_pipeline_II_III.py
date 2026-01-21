import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from typing import List, Optional, Tuple, Union
from PIL import Image
import numpy as np
import numpy as np
import pyiqa
import glob
import argparse
import shutil
from utils_score import cal_zscore
from utils_fft import * 


class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    ## Copy from diffusers
    prev_sample: torch.FloatTensor

def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    ## Copy from diffusers
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class CustomScheduler(FlowMatchEulerDiscreteScheduler):
    ## Rewrite diffusers style scheduler
    def __init__(self, vae, image_processor, device='cpu', *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.vae = vae
        self.image_processor = image_processor
        self.device = device
        self.ori_size = (0, 0)
        self.new_size = (0, 0)
        self.loss_func = torch.nn.MSELoss()
        self.similarity = torch.nn.L1Loss()

    def get_latents(self, path):
        img = Image.open(path).convert('RGB')
        self.ori_size = img.size
        w, h = img.size
        # for now we consider minimize size to 512x512
        target_w = 512 if w <= 512 else ((w + 16 - 1) // 16) * 16
        target_h = 512 if h <= 512 else ((h + 16 - 1) // 16) * 16

        img = np.pad(np.asarray(img), ((0, target_h - h), (0, target_w - w), (0, 0)), mode='constant')
        img = Image.fromarray(img)
        self.new_size = img.size
        img = self.image_processor.preprocess(img).to(torch.bfloat16)
        img = (retrieve_latents(self.vae.encode(img)) - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        img = img.to(torch.float32).detach()    
        return self.pack_latents(img, height=target_h, width=target_w)
    
    def pack_latents(self, latents, batch_size=1, num_channels_latents=16, height=512, width=512):
        height = 2 * (int(height) // (8 * 2))
        width = 2 * (int(width) // (8 * 2))
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents
    
    def get_images(self, ref, win, reject):
        self.gt_img = self.get_latents(ref)
        self.win_img = self.get_latents(win)
        self.reject_img = self.get_latents(reject)
    
    def get_ori_size(self):
        return self.ori_size
    
    def get_new_size(self):
        return self.new_size
    
    def set_hyperparameters(self, D0=0.9, T1=700, T2=100, g_scale=1000, alpha_weighting=0.5):
        self.gaussian_threshold = D0
        self.T1 = T1
        self.T2 = T2
        self.scaling_factor = g_scale
        self.weighting_parameter = alpha_weighting

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)
        
        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        model_output_type = model_output.dtype

        ## TTPO core code START ##
        gradient_cond = self.condition_func(sample, model_output, sigma, self.gt_img, timestep, self.win_img, self.reject_img)
        sample = sample - gradient_cond
        ## TTPO core code END ##

        prev_sample = sample + (sigma_next - sigma) * model_output
        prev_sample = prev_sample.to(model_output_type)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def condition_func(self, x_t, model_output, sigma, x_gt_bar, t, win, lose, D0=0.9, T1=700, T2=100, g_scale=1000, alpha_weighting=0.5):
        """
        x_gt_bar: restored image
        t: time_step_list
        win: preferred image
        lose: dispreferred image
        D0: threshold for Gaussian filter
        T1, T2: two timestep thresholds (Refer to Sec 4.3)
        g_scale: scaling factor (Eq. 10)
        alpha_weighting: weighting parameter (Eq. 8) 
        """
        with torch.enable_grad():
            x_in_ori = x_t.detach().requires_grad_(True)
            x_in = x_in_ori - sigma * model_output  ## Predict x_0 use Eq. 2

            H, W = x_in.shape[-2:]
            G = gaussian_lowpass_filter((H, W), D0=D0, device=x_in.device)
            x_in_high = apply_highpass(x_in, D0=D0, filter=G)
            x_in_low = apply_lowpass(x_in, D0=D0, filter=G)
            
            # Bradley-Terry style loss, i.e., L_TTPO
            dist_win = self.similarity(x_in_high, apply_highpass(win.detach(), D0=D0, filter=G).detach())
            dist_lose = self.similarity(x_in_high, apply_highpass(lose.detach(), D0=D0, filter=G).detach())
            delta_r = dist_lose - dist_win
            # prob = torch.sigmoid(delta_r)
            # nllloss = -torch.log(prob).mean() 
            prob = torch.sigmoid(delta_r)
            nllloss = -torch.log(prob).mean()  ## Eq. 11
            
            if t > T1: 
                l1loss = self.loss_func(x_in, x_gt_bar.detach())  ## Eq. 12, L_r
                totalloss = l1loss
            elif t > T2:
            # # else:
                l1loss = self.loss_func(x_in_low, apply_lowpass(x_gt_bar.detach(), D0=0.9, filter=G).detach())
                totalloss = nllloss * alpha_weighting + l1loss
            else:
                l1loss = self.loss_func(x_in_low, apply_lowpass(x_gt_bar.detach(), D0=0.9, filter=G).detach())
                totalloss = nllloss * alpha_weighting

            totalloss *= g_scale

            return torch.autograd.grad(totalloss, x_in_ori)[0]



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--flux_path", default="pretrained/FLUX-1-dev", help="path to your flux weights")
    parser.add_argument("--seed", default=666666, help="seed")
    parser.add_argument("--prompt", default="flux_full_p_embeds.pt", help="pre-encoded prompt")
    parser.add_argument("--pooled_prompt", default="flux_full_p_pooled_embeds.pt", help="pre-encoded pooled_prompt")
    parser.add_argument("--out_path", default="output")
    parser.add_argument("--generate_path", default="generate", help="stage I results")
    parser.add_argument("--restored_path", default="restored", help="your inputs")
    parser.add_argument("--D0", default=0.9, help="gaussian lowpass filter threshold")
    parser.add_argument("--T1", default=700, help="the first timestep threshold (Sec 4.3)")
    parser.add_argument("--T2", default=100, help="the second timestep threshold (Sec 4.3)")
    parser.add_argument("--g", default=1000, help="scaling factore for L_c (Eq. 10)")
    parser.add_argument("--alpha", default=0.5, help="weighting parameter for L_TTPO (Eq. 8)")
    parser.add_argument("--denoising_step", default=50, help="number of denoising steps")
    args = parser.parse_args()

    pipe = FluxPipeline.from_pretrained(args.flux_path, torch_dtype=torch.bfloat16, device_map='balanced', text_encoder=None, text_encoder_2=None)
    ## We do not load text encoders for saving GPU memory
    vae_decoder = pipe.vae
    image_processor = pipe.image_processor
    flux_config = dict([('num_train_timesteps', 1000), ('shift', 3.0), ('use_dynamic_shifting', True), ('base_image_seq_len', 256), ('base_shift', 0.5), ('max_shift', 0.15), ('max_image_seq_len', 4096)])
    ## Just copy from flux code

    ## Build TTPO scheduler
    pipe.scheduler = CustomScheduler(vae_decoder, image_processor, 'cuda:1', **flux_config)

    # prompt = "Cinematic, high-contrast, photo-realistic, 8k, ultra HD, meticulous detailing, hyper sharpness, perfect without deformations."
    ## This prompt is copied from InvSR paper (https://arxiv.org/abs/2412.09013v2)
    ## We pre-encode it into pt file to save GPU memory

    prompt_embeds = torch.load(args.prompt, weights_only=True)
    pooled_prompt_embeds = torch.load(args.pooled_prompt, weights_only=True)

    ## Prepare automatic metrics for Selection (Stage II)
    musiq_metric = pyiqa.create_metric('musiq', device='cuda:1')
    maniqa_metric = pyiqa.create_metric('maniqa', device='cuda:1')
    qalign_metric = pyiqa.create_metric('qalign', device='cuda:1')
    mt_list = [musiq_metric, maniqa_metric, qalign_metric]

    ## Prepare paths
    out_path = f'{args.out_path}/D0_{args.D0}_T1_{args.T1}_T2_{args.T2}_g_{args.g}_alpha_{args.alpha}'
    generate_path = args.generate_path
    restore_path = args.restored_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)


    for img_name in sorted(os.listdir(generate_path)):

        print(f"Current processing: {img_name}!")

        restored_path = f'{restore_path}/{img_name}.png'  ## (Stage I results)
        refer_path_list = sorted(glob.glob(f'{generate_path}/{img_name}/*.png'))
        refer_path_list = [p for p in refer_path_list] + [restored_path]
        print(f"Total number of candidate preference images: ", len(refer_path_list))

        win_path, lose_path = cal_zscore(refer_path_list, mt_list)
        pipe.scheduler.get_images(restored_path, win_path, lose_path)

        ori_w, ori_h = pipe.scheduler.get_ori_size()
        new_w, new_h = pipe.scheduler.get_new_size()
        print(f"Original image size: {(ori_h, ori_w)}, padding to: {(new_h, new_w)}.")

        ## Stage III
        image = pipe(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, num_inference_steps=args.denoising_step, generator=torch.manual_seed(args.seed), guidance_scale=3.5, height=new_h, width=new_w).images[0]

        image = image.crop((0, 0, ori_w, ori_h))
        image.save(f"{out_path}/{img_name}_TTPO.png")
