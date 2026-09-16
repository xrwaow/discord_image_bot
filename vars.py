from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    name: str
    model_path: str
    clip_path: Optional[str]
    vae_path: Optional[str]
    arch: str  # 'sdxl', 'zimg', 'anima', 'krea2'
    default_positive: str = ""
    default_negative: str = ""


@dataclass
class LoraConfig:
    name: str
    arch: str
    loras: List[Dict[str, float]]
    keywords: str = ""


# --- API KEYS ---
with open("api_keys.yaml", "r") as file:
    data = yaml.safe_load(file)

DISCORD_TOKEN = data["DISCORD_TOKEN"]
OPENROUTER_API_KEY = data["OPENROUTER_API_KEY"]
USER_IDS = [
    data["USER_ID"],  # xr
    607557680948576287,
    1233520824913236039,
    518074129815961620,
    1318327484223197214,
]

# --- GLOBAL TOGGLES ---
ENABLE_REIMAGINE = True
ENABLE_UPSCALE_REACTIONS = True  # TODO: add separate for weak/hard upscale
ENABLE_STANDALONE_UPSCALE = False
ENABLE_ENHANCE_PROMPT = False

# --- BOT EMOJIS & SETTINGS ---
REROLL_EMOJI = "🌺"
DELETE_EMOJI = "🗑️"
UPSCALE_WEAK_EMOJI = "🔎"
UPSCALE_HARD_EMOJI = "🎨"
NUMBER_EMOJIS = ["1️⃣", "2️⃣", "3️⃣", "4️⃣"]

SAMPLERS = ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde", "er_sde"]
SCHEDULERS = ["normal", "simple", "beta", "sgm_uniform"]

# --- MODELS CONF ---
MODELS = {
    "i32_x2_mk2_aaa": ModelConfig(
        name="i32_x2_mk2_aaa",
        model_path="i32_x2_mk2_aaa.safetensors",
        clip_path=None,
        vae_path=None,
        arch="sdxl",
        default_positive="masterpiece, best quality, absurdres, very awa",
        default_negative="worst quality, bad quality, bad anatomy, embedding:lazyneg",
    ),
    "z_image": ModelConfig(
        name="z_image",
        model_path="z_image_turbo_bf16.safetensors",
        clip_path="qwen_3_4b.safetensors",
        vae_path="ae.safetensors",
        arch="zimg",
    ),
    "anima": ModelConfig(
        name="anima",
        model_path="anima-base-v1.0.safetensors",
        clip_path="qwen_3_06b_base.safetensors",
        vae_path="qwen_image_vae.safetensors",
        arch="anima",
        default_positive="best quality, masterpiece, safe",
        default_negative="worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, lowres",
    ),
    "krea2": ModelConfig(
        name="krea2",
        model_path="krea2_turbo_int8_convrot.safetensors",
        clip_path="qwen3vl_4b_fp8_scaled.safetensors",
        vae_path="qwen_image_vae.safetensors",
        arch="krea2",
    ),
}

ACTIVE_MODEL_NAME = "anima"
ACTIVE_MODEL = MODELS[ACTIVE_MODEL_NAME]

# --- LORAS CONF ---
LORAS = [
    LoraConfig("WAI", "sdxl", [{"path": "96YOTTEA-WAI.safetensors", "strength": 0.9}]),
    LoraConfig("kaestyle", "anima", [{"path": "kae/kae_style_v2-anima.safetensors", "strength": 1}], keywords="@kaestyle"),
    LoraConfig(
        "waow",
        "sdxl",
        [
            {"path": "Ah_yes.safetensors", "strength": 0.25},
            {"path": "XXX667.safetensors", "strength": 0.5},
            {"path": "0__11Xx.safetensors", "strength": 0.5},
        ],
    ),
    LoraConfig(
        "dino",
        "sdxl",
        [
            {
                "path": "noob05-dinoartforame-jan30v2-step00002464.safetensors",
                "strength": 0.75,
            }
        ],
        keywords="dino (dinoartforame)",
    ),
    LoraConfig(
        "Nyt3_Tyd3",
        "sdxl",
        [{"path": "Nyt3_Tyd3_style.safetensors", "strength": 0.9}],
        keywords="Nyt3_Tyd3_illu",
    ),
    LoraConfig(
        "grainscape",
        "zimg",
        [{"path": "z_image/grainscape_zimage.safetensors", "strength": 1.0}],
    ),
    LoraConfig(
        "lineart",
        "zimg",
        [{"path": "z_image/line_dive_zimage_turbo_512.safetensors", "strength": 0.75}],
        keywords="line_dive, Fine-Line Ink Illustration",
    ),
    LoraConfig(
        "Sh1nStl",
        "zimg",
        [{"path": "z_image/Sh1nStl_V3.safetensors", "strength": 1.0}],
        keywords="Sh1nStl style",
    ),
]

# Filter LORAs dynamically by active arch
ACTIVE_LORAS = {l.name: l for l in LORAS if l.arch == ACTIVE_MODEL.arch}

# --- ARCHITECTURE DEFAULTS ---
ARCH_DEFAULTS = {
    "sdxl": {
        "width": 896,
        "height": 1152,
        "steps": 20,
        "cfg": 7.0,
        "batch_size": 1,
        "sampler_name": "euler",
        "scheduler": "normal",
    },
    "zimg": {
        "width": 1024,
        "height": 1280,
        "steps": 8,
        "cfg": 1.0,
        "batch_size": 1,
        "sampler_name": "euler",
        "scheduler": "simple",
    },
    "anima": {
        "width": 896,
        "height": 1152,
        "steps": 25,
        "cfg": 4.0,
        "batch_size": 1,
        "sampler_name": "er_sde",
        "scheduler": "simple",
    },
    "krea2": {
        "width": 1024,
        "height": 1024,
        "steps": 8,
        "cfg": 1.0,
        "batch_size": 1,
        "sampler_name": "euler",
        "scheduler": "simple",
    },
}

ARCH_UPSCALE_WEAK = {
    "sdxl": {
        "scale": 1.25,
        "denoising_strength": 0.4,
        "steps": 8,
        "cfg": 7.0,
        "sampler_name": "euler",
        "scheduler": "normal",
    },
    "zimg": {
        "scale": 1.25,
        "denoising_strength": 0.3,
        "steps": 4,
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": "simple",
    },
    "anima": {
        "scale": 1.25,
        "denoising_strength": 0.4,
        "steps": 12,
        "cfg": 4.0,
        "sampler_name": "er_sde",
        "scheduler": "simple",
    },
    "krea2": {
        "scale": 1.25,
        "denoising_strength": 0.25,
        "steps": 4,
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": "simple",
    },
}

ARCH_UPSCALE_HARD = {
    "sdxl": {
        "scale": 1.25,
        "denoising_strength": 0.75,
        "steps": 12,
        "cfg": 7.0,
        "sampler_name": "euler",
        "scheduler": "normal",
    },
    "zimg": {
        "scale": 1.50,
        "denoising_strength": 0.4,
        "steps": 6,
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": "simple",
    },
    "anima": {
        "scale": 1.25,
        "denoising_strength": 0.75,
        "steps": 20,
        "cfg": 4.0,
        "sampler_name": "er_sde",
        "scheduler": "simple",
    },
    "krea2": {
        "scale": 1.50,
        "denoising_strength": 0.4,
        "steps": 6,
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": "simple",
    },
}

active_txt2img_args = ARCH_DEFAULTS.get(ACTIVE_MODEL.arch, ARCH_DEFAULTS["sdxl"])
active_upscale_weak_args = ARCH_UPSCALE_WEAK.get(
    ACTIVE_MODEL.arch, ARCH_UPSCALE_WEAK["sdxl"]
)
active_upscale_hard_args = ARCH_UPSCALE_HARD.get(
    ACTIVE_MODEL.arch, ARCH_UPSCALE_HARD["sdxl"]
)

# --- KEYWORDS ---
KEYWORDS = {
    "yuri": "1girl, silver hair, long hair, red eyes, black adidas tracksuit",
    "nori": "1girl, black hair, long hair, red eyes, fox ears, brown skin",
    "hibiki": "Yukari Akiyama, girls und panzer",
    "potato": "frieren",
    "marcy": "marcy wu, amphibia, swept bangs, green hairclip",
}

WILDCARDS = {
    "place": [
        "a park",
        "a city street",
        "a shopping mall",
        "a beach",
        "a subway station",
        "a forest",
        "a library",
        "a convenience store",
        "an empty parking lot at night",
        "a cafe",
        "a school hallway",
        "a greenhouse",
        "an alleyway",
    ],
    "style": [
        "digital painting",
        "sketch art",
        "oil painting",
        "film grain 35mm",
        "concept art",
    ],
    "pose": [
        "sitting",
        "running",
        "jumping",
        "dancing",
        "leaning against a wall",
        "looking over shoulder",
        "arms crossed",
        "kneeling",
    ],
}
