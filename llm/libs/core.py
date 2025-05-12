import base64
import mimetypes
import requests # Needs installation: pip install requests

from pathlib import Path
from openai import OpenAI, AsyncOpenAI
 
 
# Assuming prompts.py exists in the same directory or PYTHONPATH
try:
    from prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_MERGE
except ImportError:
    print("Warning: Could not import SYSTEM_PROMPT from prompts.py. Using a default.")
    SYSTEM_PROMPT = "You are a helpful assistant that describes images."

class ImageDescriber:
    """Encapsulates logic for describing images using OpenAI's vision models."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4.1-nano"):
        """
        Initializes the OpenAI client.

        Args:
            api_key: Your OpenAI API key. Defaults to env variable OPENAI_API_KEY.
            model: The OpenAI model to use (e.g., "gpt-4.1-nano", "gpt-4o").
        """
        self.client = AsyncOpenAI(api_key=api_key) # Reads OPENAI_API_KEY from env if api_key is None
        self.model = model
        # Basic check for API key
        if not self.client.api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or pass it to the constructor.")

    @staticmethod
    def encode_image_to_data_uri(image_source: str | Path) -> str | None:
        """
        Encodes an image from a local file path or a web URL into a base64 data URI.

        Args:
            image_source: The path to the local image file (as string or Path object)
                          or the URL of the image on the web.

        Returns:
            A base64 data URI string (e.g., "data:image/jpeg;base64,...")
            or None if encoding fails or the source is invalid.
        """
        try:
            image_path = Path(image_source)
            if image_path.is_file():
                # Handle local file
                mime_type, _ = mimetypes.guess_type(image_path)
                if not mime_type or not mime_type.startswith('image/'):
                    print(f"Error: Could not determine image type for local file: {image_path}")
                    return None

                with open(image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return f"data:{mime_type};base64,{encoded_string}"

            elif str(image_source).startswith(("http://", "https://")):
                # Handle URL
                response = requests.get(str(image_source), stream=True)
                response.raise_for_status() # Raise an exception for bad status codes

                content_type = response.headers.get('content-type')
                if not content_type or not content_type.startswith('image/'):
                    # Try guessing from URL if header is missing/wrong
                    mime_type, _ = mimetypes.guess_type(str(image_source))
                    if not mime_type or not mime_type.startswith('image/'):
                         print(f"Error: Cannot determine image type from URL or headers: {image_source}")
                         return None
                    content_type = mime_type

                encoded_string = base64.b64encode(response.content).decode('utf-8')
                return f"data:{content_type};base64,{encoded_string}"

            else:
                print(f"Error: Invalid image source. Not a local file or valid URL: {image_source}")
                return None

        except FileNotFoundError:
            print(f"Error: Local file not found: {image_source}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {image_source}: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during image encoding: {e}")
            return None

    def merge_descriptions(self, descriptions: list[str]) -> str:
        """
        Merges a list of descriptions into a single description.
        """
        return "\n".join(descriptions)
    
    async def describe_merged_descriptions(self, descriptions: list[str], 
                                     system_prompt: str = SYSTEM_PROMPT_MERGE, 
                                     temperature: float = 0.4, 
                                     max_tokens: int = 1024, 
                                     top_p: float = 1.0) -> str:
        """
        Describes a merged description.
        """
        prompt = "Merge the following descriptions into a single description:"
        
        descriptions_str = "\n".join(descriptions)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "text",
                                "text": descriptions_str,
                            },
                        ],
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            # Extract the description text safely
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                print("Warning: Could not parse description from OpenAI response.")
                print(f"Full Response: {response}")
                return None
        except Exception as e:
            # Catch potential API errors or other issues
            print(f"Error during OpenAI API call: {e}")
            return None 

    async def describe_image(
        self,
        image_url: str,
        prompt: str = "Describe the following image:",
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.4,
        max_tokens: int = 1024, # Reduced default for potentially faster/cheaper responses
        top_p: float = 1.0
    ) -> str | None:
        """
        Sends an image and prompt to the OpenAI API for description.

        Args:
            image_url: The URL or base64 data URI of the image.
            prompt: The text prompt to accompany the image.
            system_prompt: The system message to guide the model's behavior.
            temperature: Controls randomness (0.0 to 2.0).
            max_tokens: The maximum number of tokens to generate.
            top_p: Nucleus sampling parameter.

        Returns:
            The generated description string, or None if an error occurs.
        """
        if not image_url:
            print("Error: No image URL provided.")
            return None

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    # Allow detail setting if needed, default is 'auto'
                                    "url": image_url,
                                },
                            },
                        ],
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            # Extract the description text safely
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                print("Warning: Could not parse description from OpenAI response.")
                print(f"Full Response: {response}")
                return None
        except Exception as e:
            # Catch potential API errors or other issues
            print(f"Error during OpenAI API call: {e}")
            return None