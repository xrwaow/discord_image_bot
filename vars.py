
MODEL_NAME = "z_image"
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
    "prompt": [
        "A woman standing on a rainy city sidewalk holding a transparent umbrella, making a cute pouty expression. Raindrops on the umbrella clearly visible between heads and raised arms. Her floral top appears slightly damp at the shoulders. Street signs and headlights create soft bokeh lights. Photorealistic mood.",
        "A young woman is taking a mirror selfie inside a modern stainless-steel elevator. She has long, straight dark hair that falls over her shoulders. She’s making a playful duck-face expression while looking slightly to the side instead of directly at the camera. She’s holding a dark-colored smartphone with her right hand, her elbow slightly bent. She’s wearing an off-shoulder, short-sleeved black crop top with a white floral pattern, exposing her shoulders and a bit of her midriff. She also wears high-waisted black pants. The lighting inside the elevator is bright and even, reflecting softly on the metallic walls. The elevator interior features symmetrical button panels on both sides, with rows of metallic circular buttons. The floor has a mix of white and darker tiles, and the lower part of the elevator wall has a marble-like texture. The overall vibe is casual, slightly playful, and clean, with a sense of modern everyday style.",
        "90's photo style with camera flash, small orange date digital text \"13.08.1995\" in bottom right corner, japanese mountain road, nissan silvia s-13, japanese woman wearing hoodie and pleated skirt posing in front of car, summer, sunlight",
        "Style is photograph, RAW photo, taken on a Canon camera. a young woman, posing dramatically in front of a mountain village, sun rise, there is water on the right, a cat on a roof in the background",
        "Ultra-detailed, cinematic photograph of a modern wooden building with a unique architectural design, featuring multiple rectangular panels arranged in a grid-like pattern, each panel containing a small rectangular opening. The building is located on a rocky terrain, with a body of water visible with a cinematic backdrop of. The sky is overcast and foggy, creating a hazy atmosphere. On the top of the building, there are several small green plants growing on the roof, adding a touch of greenery to the roof. In the foreground, a person wearing a dark jacket and dark pants stands facing away from the camera, with their back to the viewer. The person is standing on a gravel ground with a puddle of water in front of them, reflecting the building and the surrounding landscape. The ground is covered in dark gravel, and there is a large dark rock on the right side of A cinematic scene. A wooden platform or walkway extends from the building towards the entrance, which is made of wooden planks and has several string lights hanging from it. The entrance has a glass door that allows natural light to enter, and a small table and chairs can be seen inside the glass door. A stone wall appears vividly on the left side, partially obscured by the building's exterior. The overall lighting is soft and muted, giving A cinematic scene a peaceful and serene feel. High fidelity, realistic texture, ultra detail, cinematic tone mapping.",
        "Color film photography, heavy film grain, pronounced noise一张具有2000年代早期过期胶卷摄影美学的抓拍照片。画面呈现出明显的暖色调漏光效果，色彩严重褪色且饱和度极低，对比度差，暗部细节呈现浑浊的灰黑色。整体图像带有明显的运动模糊，模拟在移动列车上手持傻瓜相机拍摄的不稳定感，并伴有直接闪光灯造成的生硬高光和暗角。  画面主体是一位年轻的日本女性，以2000年代街头哥特风格装扮。她背靠着通勤列车内部米灰色的车厢壁站立。她留着黑色的层次感碎发，皮肤苍白，化着深色眼线，身穿黑色的层叠蕾丝连衣裙，搭配渔网袖套和带银色铆钉的皮革项圈。她手中拿着一根棒棒糖，正将其送入口中，表情显得漫不经心。  在画面的右侧，一只黑猫端正地坐在褪色的深红色天鹅绒列车座椅上，猫的姿态警觉，黄色的眼睛在闪光灯下反射出亮光。背景中可以看到模糊的不锈钢扶手杆和反射着光晕的黑暗车窗，车厢顶部悬挂着难以辨认的长方形广告牌。整张照片充满颗粒感，营造出一种陈旧、疏离且真实的复古氛围。",
    ] # this is for z_image only rn, TODO: add different prompts for anime
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

LORA_CONFIG_Z_IMAGE = {
    "Movie": [
        {
            "lora": "z_image/movie_zimage_lora.safetensors",
            "strength": 1,
        },
    ],
    "grainscape": [
        {
            "lora": "z_image/grainscape_zimage.safetensors",
            "strength": 1,
        },
    ]
}

DIMENSION_PRESETS = {
    "512x640": (512, 640),
    "896x1152": (896, 1152),
    "1024x1024": (1024, 1024),
    "1024x1280": (1024, 1280),
    "1120x1440": (1120, 1440),
    "1440x1120": (1440, 1120),
    "1536x1280": (1536, 1280),
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

    LORA_CONFIG = LORA_CONFIG_Z_IMAGE
    DEFAULT_POSITIVE_PROMPT = DEFAULT_NEGATIVE_PROMPT = ""
