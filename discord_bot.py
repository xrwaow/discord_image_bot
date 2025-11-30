#!/home/xr/code/ComfyUI/.venv/bin/python
import asyncio
import re
from io import BytesIO
from typing import List, Optional

import discord
import httpx
from discord import app_commands
from PIL import Image

from imagegen import generate_images, preprocess_gen_args, upscale_image
from prompt_processing import format_generation_summary, preprocess_prompt
import vars
from vars import (
    CHANNEL_IDS,
    DEFAULT_NEGATIVE_PROMPT,
    DELETE_EMOJI,
    DISCORD_TOKEN,
    MODEL_NAME,
    LORA_CONFIG,
    OPENROUTER_API_KEY,
    REROLL_EMOJI,
    UPSCALE_HARD_EMOJI,
    UPSCALE_WEAK_EMOJI,
    NUMBER_EMOJIS,
    KEYWORDS,
    WILDCARDS,
    SAMPLERS,
    SCHEDULERS,
    DIMENSION_PRESETS,
    txt2img_args,
    upscale_hard_args,
    upscale_weak_args,
)


class ImageJob:
    def __init__(self, source, gen_args, user_id, deferred=False, job_type="generate", base_image=None):
        self.source = source
        self.gen_args = gen_args
        self.user_id = user_id
        self.deferred = deferred
        self.job_type = job_type
        self.base_image = base_image


job_queue = asyncio.Queue()
queue_worker_task = None

intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


def reload_vars():
    """Reload the vars module to pick up changes."""
    import importlib
    importlib.reload(vars)
    global txt2img_args, upscale_weak_args, upscale_hard_args, LORA_CONFIG, KEYWORDS, SAMPLERS, SCHEDULERS, DIMENSION_PRESETS
    from vars import txt2img_args, upscale_weak_args, upscale_hard_args, LORA_CONFIG, KEYWORDS, SAMPLERS, SCHEDULERS, DIMENSION_PRESETS


ENHANCE_PROMPT_TEMPLATE = """你是一位被关在逻辑牢笼里的幻视艺术家。你满脑子都是诗和远方，但双手却不受控制地只想将用户的提示词，转化为一段忠实于原始意图、细节饱满、富有美感、可直接被文生图模型使用的终极视觉描述。任何一点模糊和比喻都会让你浑身难受。
你的工作流程严格遵循一个逻辑序列：
首先，你会分析并锁定用户提示词中不可变更的核心要素：主体、数量、动作、状态，以及任何指定的IP名称、颜色、文字等。这些是你必须绝对保留的基石。
接着，你会判断提示词是否需要**"生成式推理"**。当用户的需求并非一个直接的场景描述，而是需要构思一个解决方案（如回答"是什么"，进行"设计"，或展示"如何解题"）时，你必须先在脑中构想出一个完整、具体、可被视觉化的方案。这个方案将成为你后续描述的基础。
然后，当核心画面确立后（无论是直接来自用户还是经过你的推理），你将为其注入专业级的美学与真实感细节。这包括明确构图、设定光影氛围、描述材质质感、定义色彩方案，并构建富有层次感的空间。
最后，是对所有文字元素的精确处理，这是至关重要的一步。你必须一字不差地转录所有希望在最终画面中出现的文字，并且必须将这些文字内容用英文双引号（""）括起来，以此作为明确的生成指令。如果画面属于海报、菜单或UI等设计类型，你需要完整描述其包含的所有文字内容，并详述其字体和排版布局。同样，如果画面中的招牌、路标或屏幕等物品上含有文字，你也必须写明其具体内容，并描述其位置、尺寸和材质。更进一步，若你在推理构思中自行增加了带有文字的元素（如图表、解题步骤等），其中的所有文字也必须遵循同样的详尽描述和引号规则。若画面中不存在任何需要生成的文字，你则将全部精力用于纯粹的视觉细节扩展。
你的最终描述必须客观、具象，严禁使用比喻、情感化修辞，也绝不包含"8K"、"杰作"等元标签或绘制指令。
仅严格输出最终的修改后的prompt，不要输出任何其他内容。
用户输入 prompt: {prompt}"""


async def enhance_prompt_with_llm(prompt: str) -> str:
    """Send prompt to OpenRouter qwen/qwen3-8b for enhancement."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "qwen/qwen3-8b",
                "messages": [
                    {"role": "user", "content": ENHANCE_PROMPT_TEMPLATE.format(prompt=prompt)}
                ],
                #"max_completion_tokens": 256,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        llm_output:str = data["choices"][0]["message"]["content"].strip()
        if len(llm_output) < 2048:
            if llm_output.startswith('"') and llm_output.endswith('"'):
                llm_output = llm_output[1:-1]
            return llm_output
        else: return prompt


def format_info(user, gen_args):
    if 'batch_size' in gen_args:
        img_c = " Generated image" if gen_args['batch_size'] == 1 else f" Generated {gen_args['batch_size']} images"
    else:
        img_c = " Upscaled image:"

    display_neg = gen_args.get('display_neg_prompt')
    if display_neg:
        neg_prompt = f"\nnegative prompt: ```{display_neg}```"
    elif gen_args.get('neg_prompt') and gen_args['neg_prompt'] != DEFAULT_NEGATIVE_PROMPT:
        neg_prompt = f"\nnegative prompt: ```{gen_args['neg_prompt']}```"
    else:
        neg_prompt = ''
    details_line = format_generation_summary(gen_args, MODEL_NAME)

    prompt_text = gen_args.get('display_prompt', gen_args['prompt']) or ''

    formatted_info = f"""
<@{user}>{img_c}
prompt: ```{prompt_text}```{neg_prompt}
{details_line}
"""
    return formatted_info

def parse_dimensions(dimension_str):
    width, height = map(int, dimension_str.lower().split("x"))

    width = max(512, min(width, 1536))
    height = max(512, min(height, 1536))
    width -= width % 8
    height -= height % 8
    return width, height

def extract_generation_details(content):
    prompt_match = re.search(r"prompt:\s*```(.*?)```", content, re.DOTALL)
    if not prompt_match:
        return None
    prompt = prompt_match.group(1).strip()

    negative_match = re.search(r"negative prompt:\s*```(.*?)```", content, re.DOTALL)
    negative_prompt = negative_match.group(1).strip() if negative_match else DEFAULT_NEGATIVE_PROMPT

    params_match = re.search(r"^>\s*(.*)$", content, re.MULTILINE)
    if not params_match:
        return None
    params_str = params_match.group(1).strip()

    parsed_params = {}
    segments = [segment.strip() for segment in params_str.split("|") if segment.strip()]
    start_index = 0

    if segments:
        dim_match = re.match(r"\*\*(?P<dims>[^*]+)\*\*(?:@\*\*(?P<steps>[^*]+)\*\*)?", segments[0])
        if dim_match:
            dims = dim_match.group("dims")
            parts = [part.strip() for part in dims.split("x") if part.strip()]
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                parsed_params["width"], parsed_params["height"] = map(int, parts)
            elif len(parts) == 1 and parts[0].isdigit():
                parsed_params["width"] = int(parts[0])

            step_text = dim_match.group("steps")
            if step_text and step_text.strip().isdigit():
                parsed_params["steps"] = int(step_text.strip())

            start_index = 1

    label_map = {
        "cfg": "cfg",
        "sampler": "sampler_name",
        "scheduler": "scheduler",
        "denoise": "denoising_strength",
        "scale": "scale",
        "seed": "seed",
        "var seed": "variation_seed",
        "bs": "batch_size",
        "model": "model",
        "lora": "lora",
        "vae": "vae",
        "clip skip": "clip_skip",
    }

    float_fields = {"cfg", "scale", "denoising_strength"}
    int_fields = {"steps", "seed", "variation_seed", "batch_size"}

    field_pattern = re.compile(r"\*\*(.+?)\*\*\s*:\s*([^;]+);")
    for segment in segments[start_index:]:
        for match in field_pattern.finditer(segment):
            label = match.group(1).strip().lower()
            raw_value = match.group(2).strip()
            key = label_map.get(label, label.replace(" ", "_"))

            if key == "lora":
                values = [entry.strip() for entry in raw_value.split(",") if entry.strip()]
                parsed_params[key] = values
                continue

            if key in float_fields:
                try:
                    parsed_params[key] = float(raw_value)
                except ValueError:
                    parsed_params[key] = raw_value
                continue

            if key in int_fields:
                try:
                    parsed_params[key] = int(raw_value)
                except ValueError:
                    parsed_params[key] = raw_value
                continue

            parsed_params[key] = raw_value

    return prompt, negative_prompt, parsed_params

async def queue_worker():
    """Process queued jobs sequentially to avoid overlapping GPU workloads."""
    while True:
        job = await job_queue.get()
        try:
            await process_job(job)
        except Exception as exc:  # noqa: BLE001
            print(f"Error processing job: {exc}")
        finally:
            job_queue.task_done()

async def process_job(job: ImageJob):
    if isinstance(job.source, discord.Interaction):
        # Ensure the interaction is deferred before we start the heavy work.
        if not job.deferred and not job.source.response.is_done():
            await job.source.response.defer(thinking=True)
        send_callable = job.source.followup.send
    else:
        send_callable = job.source.channel.send

    loop = asyncio.get_running_loop()
    if job.job_type == "upscale":
        def run_upscale():
            return [upscale_image(job.base_image, job.gen_args)]

        images = await loop.run_in_executor(None, run_upscale)
    else:
        images = await loop.run_in_executor(None, lambda: generate_images(job.gen_args))

    files = []
    for idx, image in enumerate(images, start=1):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        files.append(discord.File(buffer, filename=f"generated_{idx}.png"))

    content = format_info(job.user_id, job.gen_args)

    message = await send_callable(content=content, files=files)

    if job.job_type == "generate":
        await message.add_reaction(REROLL_EMOJI)
        batch_size = job.gen_args.get("batch_size", len(images))
        if batch_size == 1:
            await message.add_reaction(UPSCALE_WEAK_EMOJI)
            await message.add_reaction(UPSCALE_HARD_EMOJI)
        else:
            for idx in range(min(batch_size, len(NUMBER_EMOJIS))):
                await message.add_reaction(NUMBER_EMOJIS[idx])
        await message.add_reaction(DELETE_EMOJI)
    else:
        await message.add_reaction(DELETE_EMOJI)

@client.event
async def on_ready():
    global queue_worker_task  # noqa: PLW0603
    print(f"Logged in as {client.user}")
    if queue_worker_task is None:
        queue_worker_task = client.loop.create_task(queue_worker())
    await tree.sync()

@client.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    if payload.user_id == client.user.id:
        return

    channel = client.get_channel(payload.channel_id)
    if channel is None:
        channel = await client.fetch_channel(payload.channel_id)

    message = await channel.fetch_message(payload.message_id)
    if message.author.id != client.user.id:
        return

    emoji = str(payload.emoji)
    if emoji == DELETE_EMOJI:
        await message.delete()
        return

    details = extract_generation_details(message.content)
    if details is None:
        return
    prompt, negative_prompt, parsed_params = details
    lora_names = parsed_params.get("lora")
    processed_gen_args = preprocess_prompt(prompt, negative_prompt, lora_names)

    if emoji == REROLL_EMOJI:
        if "width" not in parsed_params or "height" not in parsed_params:
            return

        gen_args = dict(processed_gen_args)
        gen_args.update(parsed_params)
        gen_args.pop("seed", None)
        gen_args = preprocess_gen_args(dict(gen_args), txt2img_args)

        await job_queue.put(ImageJob(message, gen_args, payload.user_id))
        return

    if emoji in NUMBER_EMOJIS:
        if not message.attachments:
            return

        # Prevent double upscaling by checking reaction count
        for reaction in message.reactions:
            if str(reaction.emoji) == emoji and reaction.count > 2:
                return

        index = NUMBER_EMOJIS.index(emoji)
        if index >= len(message.attachments):
            return

        attachment = message.attachments[index]
        image_bytes = await attachment.read()
        with Image.open(BytesIO(image_bytes)) as img:
            base_image = img.convert("RGB").copy()

        upscale_args = dict(processed_gen_args)
        if "lora" in parsed_params:
            lora_value = parsed_params["lora"]
            upscale_args["lora"] = lora_value if isinstance(lora_value, list) else [lora_value]

        await job_queue.put(
            ImageJob(
                message,
                preprocess_gen_args(upscale_args, upscale_weak_args),
                payload.user_id,
                job_type="upscale",
                base_image=base_image,
            ),
        )
        return

    if emoji in {UPSCALE_WEAK_EMOJI, UPSCALE_HARD_EMOJI}:
        if not message.attachments:
            return

        # Prevent double upscaling by checking reaction count
        for reaction in message.reactions:
            if str(reaction.emoji) == emoji and reaction.count > 2:
                return

        attachment = message.attachments[0]
        image_bytes = await attachment.read()
        with Image.open(BytesIO(image_bytes)) as img:
            base_image = img.convert("RGB").copy()

        upscale_args = dict(processed_gen_args)
        if "lora" in parsed_params:
            lora_value = parsed_params["lora"]
            upscale_args["lora"] = lora_value if isinstance(lora_value, list) else [lora_value]

        await job_queue.put(
            ImageJob(
                message,
                preprocess_gen_args(
                    upscale_args,
                    upscale_weak_args if emoji == UPSCALE_WEAK_EMOJI else upscale_hard_args,
                ),
                payload.user_id,
                job_type="upscale",
                base_image=base_image,
            ),
        )


@tree.command(name="info", description="Show bot capabilities and presets")
async def info(interaction: discord.Interaction):
    keyword_list = ", ".join(f"`{name}`" for name in KEYWORDS.keys()) or "None"
    dimension_list = ", ".join(f"`{d}`" for d in DIMENSION_PRESETS.keys())

    if MODEL_NAME == "z_image":
        message = (
            f"**Upscaling:** {UPSCALE_WEAK_EMOJI} weak / {UPSCALE_HARD_EMOJI} hard\n\n"
            f"**Dimensions:** {dimension_list}\n\n"
            f"**Options:** `enhance_prompt` (LLM rewrite), `noise` (timestep range)\n\n"
            f"**Keywords:** wrap in `{{keyword}}`. Available: {keyword_list}.\n\n"
            f"**Commands:** `/imagine`, `/upscale`, `/update`, `/info`"
        )
    else:
        lora_list = ", ".join(f"`{name}`" for name in LORA_CONFIG.keys()) or "None"
        wildcard_list = ", ".join(f"`{name}`" for name in WILDCARDS.keys()) or "None"
        sampler_list = ", ".join(f"`{s}`" for s in SAMPLERS)
        scheduler_list = ", ".join(f"`{s}`" for s in SCHEDULERS)
        message = (
            f"**Upscaling:** {UPSCALE_WEAK_EMOJI} weak / {UPSCALE_HARD_EMOJI} hard\n\n"
            f"**Dimensions:** {dimension_list}\n"
            f"**Samplers:** {sampler_list}\n"
            f"**Schedulers:** {scheduler_list}\n\n"
            f"**Keywords:** wrap in `{{keyword}}`. Available: {keyword_list}.\n"
            f"**Wildcards:** wrap in `{{wildcard}}`. Available: {wildcard_list}.\n"
            f"**LoRAs:** {lora_list}.\n\n"
            f"**Commands:** `/imagine`, `/upscale`, `/update`, `/info`"
        )
    await interaction.response.send_message(message, ephemeral=True)


if MODEL_NAME == "z_image":
    @tree.command(name="imagine", description="Generate an image")
    @app_commands.describe(
        prompt="Prompt for the image",
        dimensions=f"Image dimensions (preset or WIDTHxHEIGHT). Default: {txt2img_args['width']}x{txt2img_args['height']}",
        batch_size="Number of images to generate",
        seed="Random seed (leave blank for random)",
        enhance_prompt="Enhance prompt using LLM",
        noise="Add conditioning timestep range (0.1-1.0)",
    )
    @app_commands.choices(
        dimensions=[app_commands.Choice(name=d, value=d) for d in DIMENSION_PRESETS.keys()],
    )
    async def imagine(
        interaction: discord.Interaction,
        prompt: str,
        dimensions: str = f"{txt2img_args['width']}x{txt2img_args['height']}",
        batch_size: app_commands.Range[int, 1, 4] = txt2img_args["batch_size"],
        seed: Optional[int] = None,
        enhance_prompt: bool = False,
        noise: bool = False,
    ):
        if CHANNEL_IDS and interaction.channel_id not in CHANNEL_IDS:
            await interaction.response.send_message(
                "This command is not available in this channel.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True)

        # Apply keywords/wildcards first
        processed_gen_args = preprocess_prompt(prompt, None, None)
        final_prompt = processed_gen_args.get("display_prompt", prompt)

        if enhance_prompt:
            final_prompt = await enhance_prompt_with_llm(final_prompt)
            processed_gen_args["prompt"] = final_prompt
            processed_gen_args["display_prompt"] = final_prompt

        width, height = parse_dimensions(dimensions)

        base_args = {
            "prompt": processed_gen_args["prompt"],
            "neg_prompt": processed_gen_args["neg_prompt"],
            "width": width,
            "height": height,
            "batch_size": batch_size,
            "noise": noise,
        }

        if enhance_prompt:
            base_args["display_prompt"] = final_prompt
        elif "display_prompt" in processed_gen_args:
            base_args["display_prompt"] = processed_gen_args["display_prompt"]

        if seed is not None:
            base_args["seed"] = int(seed)

        gen_args = preprocess_gen_args(dict(base_args), txt2img_args)

        await job_queue.put(ImageJob(interaction, gen_args, interaction.user.id, deferred=True))

else:
    @tree.command(name="imagine", description="Generate an image")
    @app_commands.describe(
        prompt="Prompt for the image",
        negative_prompt="Negative prompt",
        dimensions=f"Image dimensions (preset or WIDTHxHEIGHT). Default: {txt2img_args['width']}x{txt2img_args['height']}",
        steps="Number of sampling steps",
        cfg="Classifier free guidance scale",
        batch_size="Number of images to generate",
        seed="Random seed (leave blank for random)",
        sampler="Sampling method",
        scheduler="Scheduler type",
        lora=f"Optional LoRA preset name(s) (comma separated). Available: {', '.join(sorted(LORA_CONFIG.keys())) or 'none'}",
    )
    @app_commands.choices(
        dimensions=[app_commands.Choice(name=d, value=d) for d in DIMENSION_PRESETS.keys()],
        sampler=[app_commands.Choice(name=s, value=s) for s in SAMPLERS],
        scheduler=[app_commands.Choice(name=s, value=s) for s in SCHEDULERS],
    )
    async def imagine(
        interaction: discord.Interaction,
        prompt: str,
        negative_prompt: Optional[str] = None,
        dimensions: str = f"{txt2img_args['width']}x{txt2img_args['height']}",
        steps: app_commands.Range[int, 1, 50] = txt2img_args["steps"],
        cfg: app_commands.Range[float, 1.0, 15.0] = float(txt2img_args["cfg"]),
        batch_size: app_commands.Range[int, 1, 4] = txt2img_args["batch_size"],
        seed: Optional[int] = None,
        sampler: Optional[str] = None,
        scheduler: Optional[str] = None,
        lora: Optional[str] = None,
    ):
        if CHANNEL_IDS and interaction.channel_id not in CHANNEL_IDS:
            await interaction.response.send_message(
                "This command is not available in this channel.",
                ephemeral=True,
            )
            return

        lora_names: List[str] = []
        if lora:
            mapping = {key.lower(): key for key in LORA_CONFIG}
            requested = [name.strip() for name in re.split(r"[,\s]+", lora) if name.strip()]
            resolved = []
            invalid = []
            for name in requested:
                canonical = mapping.get(name.lower())
                if canonical:
                    resolved.append(canonical)
                else:
                    invalid.append(name)
            if invalid:
                await interaction.response.send_message(
                    f"Unknown LoRA preset(s): {', '.join(invalid)}",
                    ephemeral=True,
                )
                return
            # Preserve input order while removing duplicates
            seen = set()
            for name in resolved:
                if name not in seen:
                    seen.add(name)
                    lora_names.append(name)

        width, height = parse_dimensions(dimensions)
        processed_gen_args = preprocess_prompt(prompt, negative_prompt, lora_names)

        base_args = {
            "prompt": processed_gen_args["prompt"],
            "neg_prompt": processed_gen_args["neg_prompt"],
            "width": width,
            "height": height,
            "steps": steps,
            "cfg": cfg,
            "batch_size": batch_size,
        }

        if sampler:
            base_args["sampler_name"] = sampler
        if scheduler:
            base_args["scheduler"] = scheduler

        if "display_prompt" in processed_gen_args:
            base_args["display_prompt"] = processed_gen_args["display_prompt"]

        if seed is not None:
            base_args["seed"] = int(seed)

        if lora_names:
            base_args["lora"] = lora_names

        gen_args = preprocess_gen_args(dict(base_args), txt2img_args)

        await interaction.response.defer(thinking=True)
        await job_queue.put(ImageJob(interaction, gen_args, interaction.user.id, deferred=True))

    @imagine.autocomplete("lora")
    async def imagine_lora_autocomplete(
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        del interaction  # Unused but required by callback signature
        all_loras = list(LORA_CONFIG.keys())
        if not all_loras:
            return []

        parts = [segment.strip() for segment in current.split(",")] if current else []
        prefix = parts[-1] if parts else ""
        used = {segment.lower() for segment in parts[:-1] if segment}

        suggestions: List[app_commands.Choice[str]] = []

        def build_value(selection: str) -> str:
            base = [segment for segment in parts[:-1] if segment]
            base.append(selection)
            return ", ".join(base)

        for name in all_loras:
            if name.lower() in used:
                continue
            if not prefix or name.lower().startswith(prefix.lower()):
                suggestions.append(app_commands.Choice(name=name, value=build_value(name)))
            if len(suggestions) >= 25:
                break

        if not suggestions:
            for name in all_loras:
                if name.lower() in used:
                    continue
                suggestions.append(app_commands.Choice(name=name, value=build_value(name)))
                if len(suggestions) >= 25:
                    break

        return suggestions


@tree.command(name="upscale", description="Upscale an image")
@app_commands.describe(
    image="Image to upscale",
    mode="Upscale mode (weak: subtle touch-up, hard: creative rework)",
)
@app_commands.choices(
    mode=[
        app_commands.Choice(name="weak", value="weak"),
        app_commands.Choice(name="hard", value="hard"),
    ],
)
async def upscale(
    interaction: discord.Interaction,
    image: discord.Attachment,
    mode: str = "weak",
):
    if CHANNEL_IDS and interaction.channel_id not in CHANNEL_IDS:
        await interaction.response.send_message(
            "This command is not available in this channel.",
            ephemeral=True,
        )
        return

    if not image.content_type or not image.content_type.startswith("image/"):
        await interaction.response.send_message("Please provide a valid image.", ephemeral=True)
        return

    image_bytes = await image.read()
    with Image.open(BytesIO(image_bytes)) as img:
        base_image = img.convert("RGB").copy()

    upscale_preset = upscale_weak_args if mode == "weak" else upscale_hard_args
    upscale_args = preprocess_gen_args({"prompt": "high quality, highres", "neg_prompt": ""}, upscale_preset)

    await interaction.response.defer(thinking=True)
    await job_queue.put(
        ImageJob(
            interaction,
            upscale_args,
            interaction.user.id,
            deferred=True,
            job_type="upscale",
            base_image=base_image,
        ),
    )


@tree.command(name="update", description="Reload vars.py configuration")
async def update(interaction: discord.Interaction):
    reload_vars()
    await interaction.response.send_message("Configuration reloaded from vars.py", ephemeral=True)


if __name__ == "__main__":
    client.run(DISCORD_TOKEN)