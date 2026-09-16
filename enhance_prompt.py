from dataclasses import dataclass
from typing import Dict


@dataclass
class EnhancePromptConfig:
    arch: str  # same as ModelConfig.arch
    prompt: str  # sent along the user prompt; {prompt} is replaced with it


ENHANCE_PROMPTS: Dict[str, EnhancePromptConfig] = {
    "anima": EnhancePromptConfig(
        arch="anima",
        prompt="""You are a prompt generator for text-to-image model. Convert the user's request below into a single prompt.

        Rules:
        - Start with quality/meta/safety tags: "masterpiece, best quality, score_7, safe," (adjust "safe" to "sensitive", "nsfw", or "explicit" only if the user's request clearly calls for it).
        - Follow with subject count tags if applicable (e.g. "1girl", "1boy", "1other").
        - Then character name and series, if a known character/series is referenced.
        - Prefix any artist name with @ (e.g. "@artist name"). Only include an artist tag if the user explicitly asks for a specific art style/artist.
        - Follow with general descriptive tags: lowercase, comma-separated, spaces instead of underscores (e.g. "brown hair", not "brown_hair").
        - You may mix in natural language phrases alongside tags where it helps describe pose, scene, or composition more clearly than tags alone.
        - Do not include underscores except in score tags.
        - Do not include negative prompts, explanations, or markdown formatting.
        - Respond with ONLY the final prompt as a single line of comma-separated tags/phrases, nothing else.

        User request: {prompt}""",
    ),
}


def get_enhance_config(arch: str) -> EnhancePromptConfig | None:
    return ENHANCE_PROMPTS.get(arch)
