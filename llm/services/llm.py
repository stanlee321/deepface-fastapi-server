from libs.core import ImageDescriber
from configs import settings

image_describer = None

def initialize_image_describer():
    global image_describer
    
    if image_describer is None:
        image_describer = ImageDescriber(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL
        )
    return image_describer


def describe_image(image_url: str) -> str:
    initialize_image_describer()
    return image_describer.describe_image(image_url)


def describe_merged_descriptions(descriptions: list[str]) -> str:
    initialize_image_describer()
    return image_describer.describe_merged_descriptions(descriptions)


def encode_image_to_data_uri(image_path: str) -> str:
    initialize_image_describer()
    return image_describer.encode_image_to_data_uri(image_path)