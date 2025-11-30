
MODEL_NAME = "i32_x2_mk2_aaa"#"z_image"
# "ntrMIXIllustriousXL_xiii"
# "i32_x2_mk2_aaa"
# "dreambox_v40"
# "oneObsession_v18"
MODEL_PATH = f"/home/xr/code/ComfyUI/models/checkpoints/{MODEL_NAME}.safetensors"

import yaml
with open('api_keys.yaml', 'r') as file:
    data = yaml.safe_load(file)

DISCORD_TOKEN = data['DISCORD_TOKEN']
OPENROUTER_API_KEY = data['OPENROUTER_API_KEY']
CHANNEL_IDS = [data['CHANNEL_ID']]
REROLL_EMOJI = "🌺"
DELETE_EMOJI = "🗑️"
UPSCALE_WEAK_EMOJI = "🔎"
UPSCALE_HARD_EMOJI = "🎨"
NUMBER_EMOJIS = ["1️⃣", "2️⃣", "3️⃣", "4️⃣"]

SAMPLERS = ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde"]
SCHEDULERS = ["normal", "simple", "beta", "sgm_uniform"]

DEFAULT_POSITIVE_PROMPT = "masterpiece, best quality, absurdres, very awa"
#DEFAULT_POSITIVE_PROMPT = "masterpiece, best quality, absurdres, very awa"
# "embedding:lazypos"
# "best quality"
# "masterpiece, best quality, good quality, newest"
# "masterpiece, best quality, very aesthetic, high resolution, ultra-detailed, absurdres"
# "very awa, highres,"
# masterpiece, best quality, absurdres, safe

DEFAULT_NEGATIVE_PROMPT = "worst quality, bad quality, bad anatomy, embedding:lazyneg"##embedding:lazyhand"
#DEFAULT_NEGATIVE_PROMPT = "embedding:Smooth_Negative-neg, embedding:deep_negative_pony"
# "embedding:lazyhand, embedding:lazyneg"
# ", lowres, worst quality, bad quality, simple background"
# "lowres, worst quality, bad quality, bad anatomy, sketch, jpeg artifacts, signature, watermark, old, oldest, nude, naked, sexual"

txt2img_args = {
    'width': 896,
    'height': 1152,
    'steps': 20,
    'cfg': 7,
    'batch_size': 1,
    'sampler_name': "euler",
    'scheduler': "normal",
}

upscale_weak_args = {
    'scale': 1.25,
    'denoising_strength': 0.4,
    'steps': 8,
    'cfg': 7,
    'sampler_name': "euler",
    'scheduler': "normal",
}

upscale_hard_args = {
    'scale': 1.25,
    'denoising_strength': 0.75,
    'steps': 12,
    'cfg': 7,
    'sampler_name': "euler",
    'scheduler': "normal",
}

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
    ],
    "style": [
        "digital painting",
        "watercolor illustration",
        "cinematic lighting",
        "sketch art",
    ],
    "pose": [
        "sitting",
        "running",
        "jumping",
        "dancing",
    ],
}

LORA_CONFIG = {
    "WAI": [
        {
            "lora": "96YOTTEA-WAI.safetensors",
            "strength": 0.9,
        },
    ],
    "waow": [
        {
            "lora": "Ah_yes.safetensors",
            "strength": 0.25,
        },
        {
            "lora": "XXX667.safetensors",
            "strength": 0.5,
        },
        {
            "lora": "0__11Xx.safetensors",
            "strength": 0.5,
        },
    ],
    "dino": [
        {
            "lora": "noob05-dinoartforame-jan30v2-step00002464.safetensors",
            "strength": 0.75,
            "keywords": "dino (dinoartforame)",
        },
    ],
    "Nyt3_Tyd3": [
        {
            "lora": "Nyt3_Tyd3_style.safetensors",
            "strength": 0.9,
            "keywords": "Nyt3_Tyd3_illu",
        },
    ],
    "nhl-004": [
        {
            "lora": "nhl-004.safetensors",
            "strength": 0.75,
            "keywords": "nhl-004",
        },
    ],
    "Ani2rel": [
        {
            "lora": "Anime_in_real.safetensors",
            "strength": 0.9,
            "keywords": "Ani2rel",
        },
    ],
}

LORA_CONFIG_beta = {
    "MonMon": [
        {
            "lora": "testing/mon_monmon2133.safetensors",
            "strength": 0.8,
        },
    ],
    "Firedotinc": [
        {
            "lora": "testing/firedotinc_noob-vpred1.0_prodigy-schedulefree_edm2_v4.safetensors",
            "strength": 1,
        },
    ],
    "Sketchy": [
        {
            "lora": "testing/SketchyLine1llust.safetensors",
            "strength": 0.8,
            "keywords": "SketchyLine,ExpressiveLinework",
        },
    ],
    "FKEY": [
        {
            "lora": "testing/FKEY_WAI.safetensors",
            "strength": 0.8,
        },
    ],
    "Betterline": [
        {
            "lora": "testing/BetterLine-Style.safetensors",
            "strength": 1,
        },
    ],
    "Zombie": [
        {
            "lora": "testing/Zombie-Style.safetensors",
            "strenght": 1,
        },
    ],
    "mossa": [
        {
            "lora": "testing/mossacannibalis_40k_style_illustrious_epoch_6.safetensors",
            "strength": 0.8,
            "keywords": "mossa-40k",
        },
    ]
}

LORA_CONFIG.update(LORA_CONFIG_beta)

DIMENSION_PRESETS = {
    "512x640": (512, 640),
    "896x1152": (896, 1152),
    "1024x1024": (1024, 1024),
    "1024x1280": (1024, 1280),
    "1120x1440": (1120, 1440),
    "1440x1120": (1440, 1120),
}

if MODEL_NAME == "z_image":
    txt2img_args = {
        'width': 1120,
        'height': 1440,
        'steps': 8,
        'cfg': 1,
        'batch_size': 1,
        'sampler_name': "euler",
        'scheduler': "simple",
    }
    upscale_weak_args = {
        'scale': 1.25,
        'denoising_strength': 0.3,
        'steps': 4,
        'cfg': 1,
        'sampler_name': "euler",
        'scheduler': "simple",
    }

    upscale_hard_args = {
        'scale': 1.5,
        'denoising_strength': 0.4,
        'steps': 6,
        'cfg': 1,
        'sampler_name': "euler",
        'scheduler': "simple",
    }
    DEFAULT_POSITIVE_PROMPT = DEFAULT_NEGATIVE_PROMPT = ""
