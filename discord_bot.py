#!/home/xr/code/ComfyUI/.venv/bin/python
import asyncio
import inspect
import re
import time
from io import BytesIO
from typing import List, Optional

import discord
import httpx
from discord import app_commands
from PIL import Image

import enhance_prompt
import vars
from imagegen import (
    generate_images,
    preprocess_gen_args,
    set_progress_bar_global_hook,
    upscale_image,
)
from prompt_processing import format_generation_summary, preprocess_prompt
from vars import (
    ACTIVE_LORAS,
    ACTIVE_MODEL,
    DELETE_EMOJI,
    DISCORD_TOKEN,
    DEFAULT_ENHANCE_PROMPT,
    ENABLE_REIMAGINE,
    ENABLE_STANDALONE_UPSCALE,
    ENABLE_UPSCALE_REACTIONS,
    KEYWORDS,
    NUMBER_EMOJIS,
    OPENROUTER_API_KEY,
    REROLL_EMOJI,
    SAMPLERS,
    SCHEDULERS,
    UPSCALE_HARD_EMOJI,
    UPSCALE_WEAK_EMOJI,
    USER_IDS,
    WILDCARDS,
    active_txt2img_args,
    active_upscale_hard_args,
    active_upscale_weak_args,
)


class ImageJob:
    def __init__(
        self,
        source,
        gen_args,
        user_id,
        deferred=False,
        job_type="generate",
        base_image=None,
    ):
        self.source, self.gen_args, self.user_id = source, gen_args, user_id
        self.deferred, self.job_type, self.base_image = deferred, job_type, base_image


def build_progress_bar(current: int, total: int, bar_length: int = 10) -> str:
    if total <= 0:
        return "░" * bar_length
    filled = min(int((current / total) * bar_length), bar_length)
    return f"`{'█' * filled + '░' * (bar_length - filled)}` {current}/{total}"


job_queue = asyncio.Queue()
queue_worker_task = None
client = discord.Client(
    intents=discord.Intents.default()
    | discord.Intents(message_content=True, reactions=True)
)
tree = app_commands.CommandTree(
    client,
    allowed_installs=app_commands.AppInstallationType(guild=True, user=True),
    allowed_contexts=app_commands.AppCommandContext(
        guild=True, dm_channel=True, private_channel=True
    ),
)


def reload_vars():
    import importlib

    importlib.reload(vars)
    global \
        active_txt2img_args, \
        active_upscale_weak_args, \
        active_upscale_hard_args, \
        ACTIVE_LORAS, \
        ACTIVE_MODEL
    global \
        ENABLE_REIMAGINE, \
        ENABLE_UPSCALE_REACTIONS, \
        ENABLE_STANDALONE_UPSCALE, \
        DEFAULT_ENHANCE_PROMPT
    from vars import (
        ACTIVE_LORAS,
        ACTIVE_MODEL,
        DEFAULT_ENHANCE_PROMPT,
        ENABLE_REIMAGINE,
        ENABLE_REIMAGINE,
        ENABLE_STANDALONE_UPSCALE,
        ENABLE_UPSCALE_REACTIONS,
        active_txt2img_args,
        active_upscale_hard_args,
        active_upscale_weak_args,
    )


async def enhance_prompt_with_llm(prompt: str) -> str:
    config = enhance_prompt.get_enhance_config(ACTIVE_MODEL.arch)
    if config is None:
        return prompt
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "z-ai/glm-5.3-flash",
                "provider": {
                    "order": [
                        "deepinfra/fp4",
                        "relace",
                        "morph/fp8",
                        "z-ai/fp8",
                    ],
                    "allow_fallbacks": False,
                },
                "messages": [
                    {
                        "role": "user",
                        "content": config.prompt.format(prompt=prompt),
                    }
                ],
                #"max_tokens": 1024,
                "reasoning": {
                    "effort": "low",
                    "exclude": True,
                },
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


def format_info(user, gen_args):
    img_c = (
        " Generated image"
        if gen_args.get("batch_size", 1) == 1
        else f" Generated {gen_args['batch_size']} images"
    )
    if "batch_size" not in gen_args:
        img_c = " Upscaled image"

    display_neg = gen_args.get("display_neg_prompt")
    neg_prompt = (
        f"\nnegative: ```{display_neg}```"
        if display_neg
        else f"\nnegative: ```{gen_args['neg_prompt']}```"
        if gen_args.get("neg_prompt")
        and gen_args["neg_prompt"] != ACTIVE_MODEL.default_negative
        else ""
    )

    details_line = format_generation_summary(gen_args, ACTIVE_MODEL.name)
    prompt_text = gen_args.get("display_prompt", gen_args["prompt"])[:1500] or ""

    return f"\n<@{user}>{img_c}\n```{prompt_text}```{neg_prompt}\n{details_line}\n"


def parse_dimensions(dimension_str):
    text = dimension_str.lower().strip()
    sep = "x" if "x" in text else ":" if ":" in text else None
    if sep is None:
        raise ValueError(f"Invalid dimensions: {dimension_str!r}")
    parts = [p.strip() for p in text.split(sep)]
    if len(parts) != 2 or not all(p.isdigit() for p in parts) or "0" in (parts[0], parts[1]):
        raise ValueError(f"Invalid dimensions: {dimension_str!r}")
    a, b = int(parts[0]), int(parts[1])

    def fit(value):
        value = max(512, min(int(value), 1536))
        return value - (value % 8)

    # ":" is an aspect ratio (e.g. "3:4"), sized to ~1MP; "x" is explicit WxH
    if sep == ":":
        return (
            fit(round((1_048_576 * a / b) ** 0.5 / 8) * 8),
            fit(round((1_048_576 * b / a) ** 0.5 / 8) * 8),
        )
    return fit(a), fit(b)


def extract_generation_details(content):
    prompt_match = re.search(
        r"Generated (?:image|\d+ images)\s*```(.*?)```", content, re.DOTALL
    )
    if not prompt_match:
        return None
    prompt = prompt_match.group(1).strip()

    negative_match = re.search(r"negative:\s*```(.*?)```", content, re.DOTALL)
    negative_prompt = (
        negative_match.group(1).strip()
        if negative_match
        else ACTIVE_MODEL.default_negative
    )

    params_match = re.search(r"^>\s*(.*)$", content, re.MULTILINE)
    if not params_match:
        return None

    parsed_params = {}
    segments = [
        s.strip() for s in params_match.group(1).strip().split("|") if s.strip()
    ]
    start_index = 0

    if segments:
        dim_match = re.match(
            r"\*\*(?P<dims>[^*]+)\*\*(?:@\*\*(?P<steps>[^*]+)\*\*)?", segments[0]
        )
        if dim_match:
            parts = [
                part.strip()
                for part in dim_match.group("dims").split("x")
                if part.strip()
            ]
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                parsed_params["width"], parsed_params["height"] = map(int, parts)
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
        "noise": "noise",
    }
    float_fields, int_fields = (
        {"cfg", "scale", "denoising_strength"},
        {"steps", "seed", "variation_seed", "batch_size"},
    )
    field_pattern = re.compile(r"\*\*(.+?)\*\*\s*:\s*([^;]+);")

    for segment in segments[start_index:]:
        for match in field_pattern.finditer(segment):
            label = match.group(1).strip().lower()
            raw_value = match.group(2).strip()
            key = label_map.get(label, label.replace(" ", "_"))

            if key == "lora":
                parsed_params[key] = [
                    entry.strip() for entry in raw_value.split(",") if entry.strip()
                ]
            elif key == "noise":
                parsed_params[key] = raw_value.lower() == "true"
            elif key in float_fields:
                try:
                    parsed_params[key] = float(raw_value)
                except ValueError:
                    parsed_params[key] = raw_value
            elif key in int_fields:
                try:
                    parsed_params[key] = int(raw_value)
                except ValueError:
                    parsed_params[key] = raw_value
            else:
                parsed_params[key] = raw_value

    return prompt, negative_prompt, parsed_params


async def queue_worker():
    while True:
        job = await job_queue.get()
        try:
            await process_job(job)
        except Exception as exc:
            print(f"Error processing job: {exc}")
        finally:
            job_queue.task_done()


async def process_job(job: ImageJob):
    if isinstance(job.source, discord.Interaction):
        if not job.deferred and not job.source.response.is_done():
            await job.source.response.defer(thinking=True)
        send_callable = job.source.followup.send
    else:
        send_callable = job.source.channel.send

    progress_msg = await send_callable(content="Starting.")
    progress_state = {
        "current": 0,
        "total": job.gen_args.get("steps", 20),
        "last_update": time.time(),
        "started": False,
        "dots": 1,
    }
    update_queue = asyncio.Queue()

    def progress_hook(current, total, preview, node_id=None):
        progress_state.update({"current": current, "total": total, "started": True})
        if time.time() - progress_state["last_update"] >= 1.0:
            progress_state["last_update"] = time.time()
            try:
                update_queue.put_nowait((current, total))
            except asyncio.QueueFull:
                pass

    async def update_progress():
        while True:
            try:
                current, total = await asyncio.wait_for(update_queue.get(), timeout=0.5)
                await progress_msg.edit(
                    content=f"Generating... {build_progress_bar(current, total)}"
                )
            except asyncio.TimeoutError:
                if not progress_state["started"]:
                    progress_state["dots"] = (progress_state["dots"] % 3) + 1
                    await progress_msg.edit(
                        content=f"Starting{'.' * progress_state['dots']}"
                    )
            except asyncio.CancelledError:
                break

    progress_task = asyncio.create_task(update_progress())
    loop = asyncio.get_running_loop()
    set_progress_bar_global_hook(progress_hook)

    try:
        if job.job_type == "upscale":
            images = await loop.run_in_executor(
                None, lambda: [upscale_image(job.base_image, job.gen_args)]
            )
        else:
            images = await loop.run_in_executor(
                None, lambda: generate_images(job.gen_args)
            )
    finally:
        set_progress_bar_global_hook(None)
        progress_task.cancel()

    files = []
    for idx, image in enumerate(images, start=1):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        files.append(discord.File(buffer, filename=f"generated_{idx}.png"))

    await progress_msg.edit(
        content=format_info(job.user_id, job.gen_args), attachments=files
    )

    if job.job_type == "generate":
        if ENABLE_REIMAGINE:
            await progress_msg.add_reaction(REROLL_EMOJI)

        if ENABLE_UPSCALE_REACTIONS:
            batch_size = job.gen_args.get("batch_size", len(images))
            if batch_size == 1:
                await progress_msg.add_reaction(UPSCALE_WEAK_EMOJI)
                await progress_msg.add_reaction(UPSCALE_HARD_EMOJI)
            else:
                for idx in range(min(batch_size, len(NUMBER_EMOJIS))):
                    await progress_msg.add_reaction(NUMBER_EMOJIS[idx])

    await progress_msg.add_reaction(DELETE_EMOJI)


@client.event
async def on_ready():
    global queue_worker_task
    print(f"Logged in as {client.user}")
    if queue_worker_task is None:
        queue_worker_task = client.loop.create_task(queue_worker())
    await tree.sync()


@client.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    if payload.user_id == client.user.id:
        return
    channel = client.get_channel(payload.channel_id) or await client.fetch_channel(
        payload.channel_id
    )
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

    processed_gen_args = preprocess_prompt(
        prompt, negative_prompt, parsed_params.get("lora")
    )

    if emoji == REROLL_EMOJI and ENABLE_REIMAGINE:
        if "width" not in parsed_params or "height" not in parsed_params:
            return
        gen_args = dict(processed_gen_args, **parsed_params)
        gen_args.pop("seed", None)
        await job_queue.put(
            ImageJob(
                message,
                preprocess_gen_args(gen_args, active_txt2img_args),
                payload.user_id,
            )
        )
        return

    if ENABLE_UPSCALE_REACTIONS and emoji in NUMBER_EMOJIS + [
        UPSCALE_WEAK_EMOJI,
        UPSCALE_HARD_EMOJI,
    ]:
        if not message.attachments or any(
            r.emoji == emoji and r.count > 2 for r in message.reactions
        ):
            return

        index = NUMBER_EMOJIS.index(emoji) if emoji in NUMBER_EMOJIS else 0
        if index >= len(message.attachments):
            return

        with Image.open(BytesIO(await message.attachments[index].read())) as img:
            base_image = img.convert("RGB").copy()

        upscale_args = dict(processed_gen_args)
        if "lora" in parsed_params:
            upscale_args["lora"] = (
                [parsed_params["lora"]]
                if isinstance(parsed_params["lora"], str)
                else parsed_params["lora"]
            )
        preset = (
            active_upscale_hard_args
            if emoji == UPSCALE_HARD_EMOJI
            else active_upscale_weak_args
        )

        await job_queue.put(
            ImageJob(
                message,
                preprocess_gen_args(upscale_args, preset),
                payload.user_id,
                job_type="upscale",
                base_image=base_image,
            )
        )


@tree.command(name="info", description="Show bot capabilities and presets")
async def info(interaction: discord.Interaction):
    kw_list = ", ".join(f"`{k}`" for k in KEYWORDS) or "None"
    lora_list = ", ".join(f"`{l}`" for l in ACTIVE_LORAS) or "None"
    wc_list = ", ".join(f"`{w}`" for w in WILDCARDS) or "None"

    msg = (
        f"**Model:** `{ACTIVE_MODEL.name}` ({ACTIVE_MODEL.arch})\n\n"
        f"**LoRAs:** {lora_list}\n\n"
        f"**Keywords:** wrap in `{{keyword}}`. Available: {kw_list}\n"
        f"**Wildcards:** wrap in `{{wildcard}}`. Available: {wc_list}\n\n"
        f"**Commands:** `/imagine`, `/info`, `/update`"
        + (", `/upscale`" if ENABLE_STANDALONE_UPSCALE else "")
    )

    await interaction.response.send_message(msg, ephemeral=True)


# Dynamically shape the /imagine slash command signature to cleanly hide unneeded inputs from Discord UI
async def _imagine_callback(
    interaction: discord.Interaction,
    prompt: str,
    negative_prompt: Optional[str] = None,
    dimensions: Optional[str] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    sampler: Optional[str] = None,
    scheduler: Optional[str] = None,
    lora: Optional[str] = None,
    enhance: Optional[bool] = None,
    raw: Optional[bool] = None,
    noise: Optional[bool] = None,
):
    if USER_IDS and interaction.user.id not in USER_IDS:
        return await interaction.response.send_message(
            "Command not available here.", ephemeral=True
        )

    await interaction.response.defer(thinking=True)

    try:
        width, height = parse_dimensions(
            dimensions
            or f"{active_txt2img_args['width']}x{active_txt2img_args['height']}"
        )
    except ValueError:
        return await interaction.followup.send(
            "Invalid dimensions. Use WxH (e.g. 896x1152) or a ratio (e.g. 3:4).",
            ephemeral=True,
        )

    lora_names = [n.strip() for n in lora.split(",")] if lora and lora != "none" else []
    lora_names = [n for n in lora_names if n in ACTIVE_LORAS]

    if raw:
        # Bypass all prompt processing: no keywords, wildcards, defaults or enhancement
        final_prompt = prompt.strip()
        processed = {"prompt": final_prompt, "neg_prompt": (negative_prompt or "").strip()}
        if negative_prompt:
            processed["display_neg_prompt"] = negative_prompt.strip()
    else:
        processed = preprocess_prompt(prompt, negative_prompt, lora_names)
        final_prompt = processed.get("display_prompt", prompt)

        use_enhance = DEFAULT_ENHANCE_PROMPT if enhance is None else enhance
        if use_enhance:
            final_prompt = await enhance_prompt_with_llm(final_prompt)
            processed["prompt"] = processed["display_prompt"] = final_prompt

    base_args = {
        "prompt": processed["prompt"],
        "neg_prompt": processed["neg_prompt"],
        "width": width,
        "height": height,
        "steps": steps or active_txt2img_args["steps"],
        "cfg": cfg or active_txt2img_args["cfg"],
        "batch_size": batch_size or active_txt2img_args["batch_size"],
        "sampler_name": sampler or active_txt2img_args["sampler_name"],
        "scheduler": scheduler or active_txt2img_args["scheduler"],
    }
    if noise is not None:
        base_args["noise"] = noise

    if "display_prompt" in processed:
        base_args["display_prompt"] = processed["display_prompt"]
    if seed is not None:
        base_args["seed"] = int(seed)
    if lora_names:
        base_args["lora"] = lora_names

    await job_queue.put(
        ImageJob(
            interaction,
            preprocess_gen_args(base_args, active_txt2img_args),
            interaction.user.id,
            deferred=True,
        )
    )


# 1. Modify the Python signature dynamically based on current configuration
sig = inspect.signature(_imagine_callback)
params = list(sig.parameters.values())
if not ACTIVE_LORAS:
    params = [p for p in params if p.name != "lora"]
if enhance_prompt.get_enhance_config(ACTIVE_MODEL.arch) is None:
    params = [p for p in params if p.name != "enhance"]
_imagine_callback.__signature__ = sig.replace(parameters=params)

# 2. Assign descriptions
desc_map = {
    "prompt": "Prompt for the image",
    "negative_prompt": "Negative prompt",
    "dimensions": "WxH dimensions or W:H ratio (ratio gives ~1MP image)",
    "steps": "Sampling steps",
    "cfg": "CFG scale",
    "batch_size": "Batch size",
    "seed": "Random seed",
    "sampler": "Sampling method",
    "scheduler": "Scheduler type",
    "noise": "Add conditioning timestep range",
    "raw": "Use prompt as-is (no keywords, wildcards, defaults or enhancement)",
}
if ACTIVE_LORAS:
    desc_map["lora"] = "Optional LoRA preset name(s)"
if enhance_prompt.get_enhance_config(ACTIVE_MODEL.arch) is not None:
    desc_map["enhance"] = "Enhance prompt using LLM"
cmd = app_commands.describe(**desc_map)(_imagine_callback)

# 3. Assign Choices
choice_map = {
    "sampler": [app_commands.Choice(name=s, value=s) for s in SAMPLERS],
    "scheduler": [app_commands.Choice(name=s, value=s) for s in SCHEDULERS],
}
if ACTIVE_LORAS:
    choice_map["lora"] = [
        app_commands.Choice(name=n, value=n) for n in list(ACTIVE_LORAS.keys())[:25]
    ]
cmd = app_commands.choices(**choice_map)(cmd)

# 4. Attach to Discord Tree
tree.command(name="imagine", description="Generate an image")(cmd)


if ENABLE_STANDALONE_UPSCALE:

    @tree.command(name="upscale", description="Upscale an image")
    @app_commands.choices(
        mode=[
            app_commands.Choice(name="weak", value="weak"),
            app_commands.Choice(name="hard", value="hard"),
        ]
    )
    async def upscale(
        interaction: discord.Interaction, image: discord.Attachment, mode: str = "weak"
    ):
        if USER_IDS and interaction.user.id not in USER_IDS:
            return await interaction.response.send_message(
                "Not available here.", ephemeral=True
            )
        if not image.content_type or not image.content_type.startswith("image/"):
            return await interaction.response.send_message(
                "Provide a valid image.", ephemeral=True
            )

        with Image.open(BytesIO(await image.read())) as img:
            base_image = img.convert("RGB").copy()
        args = preprocess_gen_args(
            {"prompt": "high quality, highres", "neg_prompt": ""},
            active_upscale_weak_args if mode == "weak" else active_upscale_hard_args,
        )

        await interaction.response.defer(thinking=True)
        await job_queue.put(
            ImageJob(
                interaction,
                args,
                interaction.user.id,
                deferred=True,
                job_type="upscale",
                base_image=base_image,
            )
        )


@tree.command(name="update", description="Reload vars.py configuration")
async def update(interaction: discord.Interaction):
    reload_vars()
    await interaction.response.send_message(
        "Configuration reloaded from vars.py", ephemeral=True
    )


if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
