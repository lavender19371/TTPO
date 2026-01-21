import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusion3Img2ImgPipeline, FluxImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline
import glob
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model1_path", default="pretrained/SD21", help="path to your SD2.1 weights")
parser.add_argument("--model2_path", default="pretrained/SD3", help="path to your SD3 weights")
parser.add_argument("--model3_path", default="pretrained/FLUX-1-dev", help="path to your flux weights")
parser.add_argument("--seed", default=666666, help="seed")
parser.add_argument("--prompt", default="flux_full_p_embeds.pt", help="pre-encoded prompt")
parser.add_argument("--pooled_prompt", default="flux_full_p_pooled_embeds.pt", help="pre-encoded pooled_prompt")
parser.add_argument("--out_path", default="output")
parser.add_argument("--restored_path", default="restored", help="your inputs")
args = parser.parse_args()


pipeSD21 = StableDiffusionImg2ImgPipeline.from_pretrained(args.model1_path, torch_dtype=torch.float16)
pipeSD21 = pipeSD21.to("cuda")
pipeSD3 = StableDiffusion3Img2ImgPipeline.from_pretrained(
    args.model2_path, 
    torch_dtype=torch.float16
    )
pipeSD3 = pipeSD3.to("cuda")
pipeFLUX = FluxImg2ImgPipeline.from_pretrained(args.model3_path, torch_dtype=torch.bfloat16)
pipeFLUX = pipeFLUX.to("cuda")
## If you have GPU device with memory larger than 24G, i.e., 48G, 80G, etc., you can load all models to one single GPU.

image_list = sorted(glob.glob(os.path.join(args.restored_path, "*.png")))
for image_path in image_list:
    image_name, ext = os.path.splitext(os.path.basename(image_path))
    out_path = os.path.join(args.out_path, image_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        print(f"skip: {image_name}")
        continue
    init_image = Image.open(image_path).convert("RGB")
    w, h = init_image.size
    target_w = 512 if w <= 512 else ((w + 16 - 1) // 16) * 16
    target_h = 512 if h <= 512 else ((h + 16 - 1) // 16) * 16
    init_image = np.pad(np.asarray(init_image), ((0, target_h - h), (0, target_w - w), (0, 0)), mode='constant')
    init_image = Image.fromarray(init_image)
    print(init_image.size)

    prompt = "Cinematic, high-contrast, photo-realistic, 8k, ultra HD, meticulous detailing, hyper sharpness, perfect without deformations."
    ## This prompt is copied from InvSR paper (https://arxiv.org/abs/2412.09013v2)

    strength_list = [0.1, 0.15, 0.2, 0.25, 0.3]

    # 1. SD2.1
    for s in strength_list:
        image = pipeSD21(prompt=prompt, image=init_image, strength=s, height=target_h, width=target_w, generator=torch.manual_seed(args.seed)).images[0]
        image = image.crop((0, 0, w, h))
        image.save(f"{out_path}/SD2-1_s{s}.png")

    # 2. SD3
    for s in strength_list:
        image = pipeSD3(prompt=prompt, image=init_image, strength=s, height=target_h, width=target_w, generator=torch.manual_seed(args.seed)).images[0]
        image = image.crop((0, 0, w, h))
        image.save(f"{out_path}/SD3_s{s}.png")

    # 3. FLUX
    for s in strength_list:
        image = pipeFLUX(prompt=prompt, image=init_image, strength=s, height=target_h, width=target_w, generator=torch.manual_seed(args.seed), num_inference_steps=50).images[0]
        image = image.crop((0, 0, w, h))
        image.save(f"{out_path}/FLUX_s{s}.png")

