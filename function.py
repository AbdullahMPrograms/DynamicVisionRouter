import os
import json
import time
import logging
import requests
import aiohttp
import google.generativeai as genai
from typing import (
    List,
    Union,
    Generator,
    Iterator,
    Dict,
    Optional,
    AsyncIterator,
    Tuple,
    Any,
)
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message


class ModelConfig(BaseModel):
    name: str
    context_length: int
    alias: Optional[str] = None


class Pipe:
    SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB per image
    TOTAL_MAX_IMAGE_SIZE = 100 * 1024 * 1024  # 100MB total
    REQUEST_TIMEOUT = (3.05, 60)

    class Valves(BaseModel):
        OPENAI_API_URL: str = Field(
            default=os.getenv("OPENAI_API_URL", ""),
            description="Your OpenAI-compatible API endpoint URL",
        )
        OPENAI_API_KEY: str = Field(
            default=os.getenv("OPENAI_API_KEY", ""),
            description="Your OpenAI-compatible API key",
        )
        GOOGLE_API_KEY: str = Field(
            default=os.getenv("GOOGLE_API_KEY", ""),
            description="Your Google API key for image processing",
        )
        CONTEXT_LENGTH: int = Field(
            default=int(os.getenv("CONTEXT_LENGTH", 32768)),
            description="Maximum context window size for the model",
        )
        NUM_PREDICT: int = Field(
            default=int(os.getenv("NUM_PREDICT", 32768)),
            description="Maximum number of tokens to generate (max_tokens/num_predict)",
        )
        TEMPERATURE: float = Field(
            default=float(os.getenv("TEMPERATURE", 0.8)),
            description="Default temperature parameter for sampling",
            ge=0.0,
            le=2.0,
        )
        OPENAI_MODELS: str = Field(
            default=os.getenv("OPENAI_MODELS", "all"),
            description="Names of the OpenAI-compatible models to use (comma-separated) or 'all' to use all available models",
        )
        GEMINI_MODEL_NAME: str = Field(
            default=os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash-lite"),
            description="Name of the Google Gemini model to use for image processing",
        )
        IMAGE_DESCRIPTION_PROMPT: str = Field(
            default=os.getenv(
                "IMAGE_DESCRIPTION_PROMPT",
                "Give a clear and detailed description of this image.",
            ),
            description="The prompt to use when asking Gemini to describe an image",
        )
        SHOW_GEMINI_DESCRIPTIONS: bool = Field(
            default=os.getenv("SHOW_GEMINI_DESCRIPTIONS", "false").lower() == "true",
            description="Show Gemini's raw image descriptions in a collapsible section",
        )
        GEMINI_SECTION_TITLE: str = Field(
            default=os.getenv("GEMINI_SECTION_TITLE", "Image Description"),
            description="Title for the collapsible Gemini descriptions section",
        )

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.type = "manifold"
        self.id = "openai-compatible"
        self.valves = self.Valves()
        self.request_id = None
        self.current_gemini_descriptions = []

    def _fetch_openai_models(self) -> List[Dict[str, Any]]:
        """Fetch available models from the OpenAI API"""
        try:
            # Determine the base URL and build the models endpoint URL
            api_base = self.valves.OPENAI_API_URL
            if "/v1/chat/completions" in api_base:
                api_base = api_base.replace("/v1/chat/completions", "")
            elif "/chat/completions" in api_base:
                api_base = api_base.replace("/chat/completions", "")

            models_url = f"{api_base.rstrip('/')}/v1/models"

            headers = {
                "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }

            response = requests.get(
                models_url, headers=headers, timeout=self.REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
            else:
                logging.error(
                    f"Failed to fetch models: HTTP {response.status_code}: {response.text}"
                )
                return []
        except Exception as e:
            logging.error(f"Error fetching OpenAI models: {str(e)}")
            return []

    def get_available_models(self) -> List[dict]:
        """Get available models based on configuration"""
        models_to_use = []

        if self.valves.OPENAI_MODELS.lower() == "all":
            # Fetch and return all models from the API
            all_models = self._fetch_openai_models()
            for model in all_models:
                models_to_use.append(
                    {
                        "id": f"{self.id}/{model.get('id')}",
                        "name": model.get("id"),
                        "context_length": self.valves.CONTEXT_LENGTH,
                        "supports_vision": True,  # Assuming all models support vision - may need refinement
                    }
                )
        else:
            # Return specified models (comma-separated)
            model_names = [
                name.strip() for name in self.valves.OPENAI_MODELS.split(",")
            ]
            for model_name in model_names:
                if model_name:
                    models_to_use.append(
                        {
                            "id": f"{self.id}/{model_name}",
                            "name": model_name,
                            "context_length": self.valves.CONTEXT_LENGTH,
                            "supports_vision": True,
                        }
                    )

        return models_to_use

    def pipes(self) -> List[dict]:
        return self.get_available_models()

    def extract_images_and_text(self, message: Dict) -> Tuple[List[Dict], str]:
        """Extract images and text from a message."""
        images = []
        text_parts = []
        content = message.get("content", "")

        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    images.append(item)
        else:
            text_parts.append(content)

        return images, " ".join(text_parts)

    async def process_image_with_gemini(
        self, image_data: Dict, __event_emitter__=None
    ) -> str:
        """Process a single image with Gemini and return its description."""
        processing_message = None
        try:
            if not self.valves.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is required for image processing")

            image_url = image_data.get("image_url", {}).get("url", "")

            if __event_emitter__:
                processing_message = {
                    "type": "status",
                    "data": {
                        "description": "Processing new image...",
                        "done": False,
                    },
                }
                await __event_emitter__(processing_message)

            genai.configure(api_key=self.valves.GOOGLE_API_KEY)
            model = genai.GenerativeModel(self.valves.GEMINI_MODEL_NAME)

            if image_url.startswith("data:image"):
                image_data = image_url.split(",", 1)[1] if "," in image_url else ""
                image_part = {
                    "inline_data": {"mime_type": "image/jpeg", "data": image_data}
                }
            else:
                image_part = {"image_url": image_url}

            # Use the configurable prompt from valves instead of hard-coded text
            prompt = self.valves.IMAGE_DESCRIPTION_PROMPT
            response = model.generate_content([prompt, image_part])
            description = response.text

            # Store the description for display if enabled
            if self.valves.SHOW_GEMINI_DESCRIPTIONS:
                self.current_gemini_descriptions.append(
                    {
                        "description": description,
                    }
                )

            return description

        except Exception as e:
            error_msg = f"[Error processing image: {str(e)}]"
            logging.error(f"Error processing image with Gemini: {str(e)}")

            # Store error info if enabled
            if self.valves.SHOW_GEMINI_DESCRIPTIONS:
                self.current_gemini_descriptions.append(
                    {"description": error_msg, "error": True}
                )

            return error_msg
        finally:
            if __event_emitter__ and processing_message:
                # Clear the processing message by sending a completion update
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            # Empty description clears the text
                            "description": "",
                            # Done flag removes the status indicator
                            "done": True,
                        },
                    }
                )

    async def process_messages(
        self, messages: List[Dict], __event_emitter__=None
    ) -> List[Dict]:
        """Process messages, replacing images with their descriptions."""
        processed_messages = []

        # Clear current descriptions
        self.current_gemini_descriptions = []

        # Find the last user message with images (that's the only one we'll process with Gemini)
        last_user_msg_with_images_idx = -1
        for i in range(len(messages) - 1, -1, -1):  # Iterate backward
            if messages[i].get("role") == "user":
                images, _ = self.extract_images_and_text(messages[i])
                if images:
                    last_user_msg_with_images_idx = i
                    break

        # Process all messages (ensure all messages are converted to text format)
        for i, message in enumerate(messages):
            images, text = self.extract_images_and_text(message)

            if images:
                if i == last_user_msg_with_images_idx:
                    # This is the latest user message with images - process them with Gemini
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": f"Found {len(images)} new image(s) to process",
                                    "done": False,
                                },
                            }
                        )

                    image_descriptions = []
                    for idx, image in enumerate(images, 1):
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": f"Processing image {idx} of {len(images)}",
                                        "done": False,
                                    },
                                }
                            )
                        description = await self.process_image_with_gemini(
                            image, __event_emitter__
                        )
                        image_descriptions.append(f"[Image Description: {description}]")

                    combined_content = text + " " + " ".join(image_descriptions)
                    processed_messages.append(
                        {"role": message["role"], "content": combined_content.strip()}
                    )
                else:
                    # For older messages with images, convert to text format but don't process with Gemini
                    # Just append placeholders for images
                    image_placeholders = [f"[Image {i+1}]" for i in range(len(images))]
                    combined_content = text + " " + " ".join(image_placeholders)
                    processed_messages.append(
                        {"role": message["role"], "content": combined_content.strip()}
                    )
            else:
                # Messages with no images can be passed through as-is
                # If the content is a list, convert it to text for compatibility
                if isinstance(message.get("content"), list):
                    # Extract only text parts
                    text_content = " ".join(
                        [
                            part.get("text", "")
                            for part in message.get("content", [])
                            if part.get("type") == "text"
                        ]
                    )
                    processed_messages.append(
                        {"role": message["role"], "content": text_content}
                    )
                else:
                    processed_messages.append(message)

        return processed_messages

    def format_gemini_descriptions(self):
        """Format Gemini descriptions as a collapsible HTML section"""
        if not self.current_gemini_descriptions:
            return ""

        # Opening details tag with configurable summary
        result = f"<details>\n<summary>{self.valves.GEMINI_SECTION_TITLE}</summary>\n\n"

        # Content inside the details tag
        for idx, desc in enumerate(self.current_gemini_descriptions, 1):
            error_prefix = "ERROR: " if desc.get("error") else ""

            result += f"**Image {idx}**\n\n"
            result += f"{error_prefix}{desc['description']}\n\n"

        # Add separator inside the details tag before closing - using Markdown's horizontal rule
        result += "---\n\n"

        # Close the details tag
        result += "</details>\n\n"
        return result

    async def pipe(
        self, body: Dict, __event_emitter__=None
    ) -> Union[str, Generator, Iterator]:
        if not all([self.valves.OPENAI_API_URL, self.valves.OPENAI_API_KEY]):
            error_msg = "Error: OPENAI_API_URL and OPENAI_API_KEY are required"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return {"content": error_msg, "format": "text"}

        try:
            # Extract the model from the request path
            requested_model = body.get("model", "").split("/")[-1]
            # If not provided, use the first model from our configured models
            if not requested_model:
                available_models = self.get_available_models()
                if available_models:
                    requested_model = available_models[0]["name"]
                else:
                    # Fallback to the first model in the OPENAI_MODELS list
                    requested_model = self.valves.OPENAI_MODELS.split(",")[0].strip()

            system_message, messages = pop_system_message(body.get("messages", []))
            processed_messages = await self.process_messages(
                messages, __event_emitter__
            )

            # Re-add system message if it exists
            if system_message:
                processed_messages.insert(0, system_message)

            payload = {
                "model": requested_model,
                "messages": processed_messages,
                "max_tokens": min(
                    body.get(
                        "num_predict", body.get("max_tokens", self.valves.NUM_PREDICT)
                    ),
                    self.valves.NUM_PREDICT,
                ),
                "temperature": float(body.get("temperature", self.valves.TEMPERATURE)),
                "top_p": (
                    float(body.get("top_p", 0.9))
                    if body.get("top_p") is not None
                    else None
                ),
                "stream": body.get("stream", False),
            }

            # Add optional parameters if they exist in the request
            if "top_k" in body:
                payload["top_k"] = int(body["top_k"])
            if "presence_penalty" in body:
                payload["presence_penalty"] = float(body["presence_penalty"])
            if "frequency_penalty" in body:
                payload["frequency_penalty"] = float(body["frequency_penalty"])

            payload = {k: v for k, v in payload.items() if v is not None}
            headers = {
                "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }

            if payload["stream"]:
                gemini_desc = ""
                if (
                    self.valves.SHOW_GEMINI_DESCRIPTIONS
                    and self.current_gemini_descriptions
                ):
                    gemini_desc = self.format_gemini_descriptions()

                return self._stream_response(
                    url=self.valves.OPENAI_API_URL,
                    headers=headers,
                    payload=payload,
                    gemini_descriptions=gemini_desc,
                    __event_emitter__=__event_emitter__,
                )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.valves.OPENAI_API_URL, headers=headers, json=payload
                ) as response:
                    if response.status != 200:
                        error_msg = (
                            f"Error: HTTP {response.status}: {await response.text()}"
                        )
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": error_msg,
                                        "done": True,
                                    },
                                }
                            )
                        return {"content": error_msg, "format": "text"}

                    result = await response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        response_text = result["choices"][0]["message"]["content"]

                        # If enabled, prepend Gemini descriptions to the response
                        if (
                            self.valves.SHOW_GEMINI_DESCRIPTIONS
                            and self.current_gemini_descriptions
                        ):
                            response_text = (
                                self.format_gemini_descriptions() + response_text
                            )

                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": "Request completed successfully",
                                        "done": True,
                                    },
                                }
                            )
                        return response_text
                    return ""

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return {"content": error_msg, "format": "text"}

    async def _stream_response(
        self,
        url: str,
        headers: dict,
        payload: dict,
        gemini_descriptions: str = "",
        __event_emitter__=None,
    ) -> AsyncIterator[str]:
        try:
            # First yield the Gemini descriptions if we have them
            if gemini_descriptions:
                yield gemini_descriptions

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_msg = (
                            f"Error: HTTP {response.status}: {await response.text()}"
                        )
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {"description": error_msg, "done": True},
                                }
                            )
                        yield error_msg
                        return

                    async for line in response.content:
                        if line and line.startswith(b"data: "):
                            try:
                                line_text = line[6:].decode("utf-8")
                                if line_text.strip() == "[DONE]":
                                    if __event_emitter__:
                                        await __event_emitter__(
                                            {
                                                "type": "status",
                                                "data": {
                                                    "description": "Request completed",
                                                    "done": True,
                                                },
                                            }
                                        )
                                    break

                                data = json.loads(line_text)
                                if (
                                    "choices" in data
                                    and len(data["choices"]) > 0
                                    and "delta" in data["choices"][0]
                                    and "content" in data["choices"][0]["delta"]
                                ):
                                    yield data["choices"][0]["delta"]["content"]
                            except json.JSONDecodeError as e:
                                logging.error(
                                    f"Failed to parse streaming response: {e}"
                                )
                                continue

        except Exception as e:
            error_msg = f"Stream error: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            yield error_msg
