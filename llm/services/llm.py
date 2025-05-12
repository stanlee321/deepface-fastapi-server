from libs.core import ImageDescriber
from configs import settings

image_describer = None

def initialize_image_describer():
    global image_describer
    
    if image_describer is None:
        image_describer = ImageDescriber(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            max_tokens=settings.OPENAI_MAX_TOKENS
        )
    return image_describer


async def describe_image(image_url: str) -> str:
    initialize_image_describer()
    return await image_describer.describe_image(image_url)


async def describe_merged_descriptions(descriptions: list[str]) -> str:
    initialize_image_describer()
    return await image_describer.describe_merged_descriptions(descriptions)
