import os
import random
import sys
from typing import Iterable, List, Optional

import numpy as np
import torch
from PIL import Image

from vars import (
    LORA_CONFIG,
    MODEL_NAME,
    MODEL_PATH,
)

COMFY_PATH = "/home/xr/code/ComfyUI/"
if COMFY_PATH not in sys.path:
    sys.path.append(COMFY_PATH)

import folder_paths  # type: ignore
from nodes import NODE_CLASS_MAPPINGS  # type: ignore
from comfy_extras.nodes_sd3 import EmptySD3LatentImage as EmptySD3LatentImageClass
from comfy.utils import set_progress_bar_global_hook

CheckpointLoaderSimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
LoraLoader = NODE_CLASS_MAPPINGS["LoraLoader"]()
VAEEncode = NODE_CLASS_MAPPINGS["VAEEncode"]()
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
EmptySD3LatentImage = EmptySD3LatentImageClass()
ConditioningSetTimestepRange = NODE_CLASS_MAPPINGS["ConditioningSetTimestepRange"]()


def _resolve_checkpoint() -> str:
    candidate = os.path.basename(MODEL_PATH) if MODEL_PATH else f"{MODEL_NAME}.safetensors"
    try:
        folder_paths.get_full_path("checkpoints", candidate)
        return candidate
    except Exception:  # noqa: BLE001 - fallback to explicit path if lookup fails
        return MODEL_PATH or candidate


def _apply_loras(model, clip, lora_keys: Iterable[str]):
    for key in lora_keys:
        for entry in LORA_CONFIG.get(key, []):
            lora_name = entry.get("lora")
            if not lora_name:
                continue
            strength = float(entry.get("strength", entry.get("strenght", 0.75)))
            model, clip = LoraLoader.load_lora(model, clip, lora_name, strength, strength)
    return model, clip


def _decoded_batch_to_pil(decoded_batch: torch.Tensor) -> List[Image.Image]:
    batch_np = decoded_batch.detach().cpu().numpy()
    batch_np = np.clip(batch_np, 0.0, 1.0)
    batch_np = (batch_np * 255).astype(np.uint8)
    return [Image.fromarray(sample) for sample in batch_np]


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    np_image = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    np_image = np.expand_dims(np_image, axis=0)
    return torch.from_numpy(np_image)


def preprocess_gen_args(gen_args, default_args):
    assert 'prompt' in gen_args and 'neg_prompt' in gen_args

    for key, value in default_args.items():
        gen_args.setdefault(key, value)

    if 'seed' not in gen_args or gen_args['seed'] is None:
        gen_args['seed'] = random.randint(0, 2**32 - 1)

    if 'lora' in gen_args:
        value = gen_args['lora']
        names = [value] if isinstance(value, str) else list(value)
        filtered = tuple(name for name in names if name)
        if filtered:
            gen_args['lora'] = filtered
        else:
            gen_args.pop('lora')

    return gen_args


def _prepare_model(lora_keys: Optional[List[str]]):
    if MODEL_NAME == "z_image":
        model = UNETLoader.load_unet("z_image_turbo_bf16.safetensors", "default")[0]
        #clip = CLIPLoader.load_clip("qwen_3_4b.safetensors", "stable_diffusion", "default")[0]
        clip = CLIPLoader.load_clip("qwen3_4b_fp8_scaled.safetensors", "stable_diffusion", "default")[0]
        vae = VAELoader.load_vae("ae.safetensors")[0]
    else:
        model, clip, vae = CheckpointLoaderSimple.load_checkpoint(_resolve_checkpoint())[:3]
    if lora_keys:
        model, clip = _apply_loras(model, clip, lora_keys)
    return model, clip, vae


def generate_images(gen_args):
    batch_size = gen_args.get('batch_size', 1)
    sampler_name = gen_args.get('sampler_name', 'euler')
    scheduler = gen_args.get('scheduler', 'normal')
    base_seed = gen_args.get('seed', random.randint(0, 2**32 - 1))
    use_noise = gen_args.get('noise', False)

    with torch.inference_mode():
        model, clip, vae = _prepare_model(gen_args.get('lora'))
        positive = CLIPTextEncode.encode(clip, gen_args['prompt'])[0]
        negative = CLIPTextEncode.encode(clip, gen_args['neg_prompt'])[0]

        if MODEL_NAME == "z_image" and use_noise:
            positive = ConditioningSetTimestepRange.set_range(positive, 0.1, 1.0)[0]
            negative = ConditioningSetTimestepRange.set_range(negative, 0.1, 1.0)[0]

        if MODEL_NAME == "z_image":
            latent = EmptySD3LatentImage.generate(
                gen_args['width'],
                gen_args['height'],
                batch_size,
            )[0]
        else:
            latent = EmptyLatentImage.generate(
                gen_args['width'],
                gen_args['height'],
                batch_size=batch_size,
            )[0]
        sampled = KSampler.sample(
            model=model,
            seed=base_seed,
            steps=gen_args['steps'],
            cfg=gen_args['cfg'],
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise=1.0,
        )[0]
        decoded = VAEDecode.decode(vae, sampled)[0]
        return _decoded_batch_to_pil(decoded)[:batch_size]

    return []


def img2img(image, gen_args):
    sampler_name = gen_args.get('sampler_name', 'euler')
    scheduler = gen_args.get('scheduler', 'normal')
    seed = gen_args.get('seed', random.randint(0, 2**32 - 1))

    with torch.inference_mode():
        model, clip, vae = _prepare_model(gen_args.get('lora'))
        positive = CLIPTextEncode.encode(clip, gen_args['prompt'])[0]
        negative = CLIPTextEncode.encode(clip, gen_args['neg_prompt'])[0]
        latent = VAEEncode.encode(vae, _pil_to_tensor(image))[0]

        sampled = KSampler.sample(
            model=model,
            seed=seed,
            steps=gen_args['steps'],
            cfg=gen_args['cfg'],
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise=gen_args['denoising_strength'],
        )[0]
        decoded = VAEDecode.decode(vae, sampled)[0]
        result_images = _decoded_batch_to_pil(decoded)

    return result_images[0]


def upscale_image(image, gen_args):
    upscaled_width = int(image.size[0] * gen_args['scale'])
    upscaled_height = int(image.size[1] * gen_args['scale'])

    upscaled_width -= upscaled_width % 16
    upscaled_height -= upscaled_height % 16

    resized_image = image.resize((upscaled_width, upscaled_height), Image.LANCZOS)

    return img2img(resized_image, gen_args)