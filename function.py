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


class CacheEntry:
    def __init__(self, description: str):
        self.description = description
        self.timestamp = time.time()


class ModelConfig(BaseModel):
    name: str
    context_length: int
    alias: Optional[str] = None


class Pipe:
    SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB per image
    TOTAL_MAX_IMAGE_SIZE = 100 * 1024 * 1024  # 100MB total
    REQUEST_TIMEOUT = (3.05, 60)
    CACHE_EXPIRATION = 30 * 60  # 30 minutes in seconds
    MODEL_LIST_CACHE_EXPIRATION = 10 * 60  # 10 minutes in seconds

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
            default=int(os.getenv("CONTEXT_LENGTH", 4096)),
            description="Maximum context window size for the model",
        )
        NUM_PREDICT: int = Field(
            default=int(os.getenv("NUM_PREDICT", 2048)),
            description="Maximum number of tokens to generate (max_tokens/num_predict)",
        )
        TEMPERATURE: float = Field(
            default=float(os.getenv("TEMPERATURE", 0.6)),
            description="Default temperature parameter for sampling",
            ge=0.0,
            le=2.0,
        )
        OPENAI_MODELS: str = Field(
            default=os.getenv("OPENAI_MODELS", "Vision Routed LLM"),
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

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.type = "manifold"
        self.id = "openai-compatible"
        self.valves = self.Valves()
        self.request_id = None
        self.image_cache = {}
        self.model_list_cache = {"models": [], "timestamp": 0}

    def clean_expired_cache(self):
        """Remove expired entries from cache"""
        current_time = time.time()
        expired_keys = [
            key
            for key, entry in self.image_cache.items()
            if current_time - entry.timestamp > self.CACHE_EXPIRATION
        ]
        for key in expired_keys:
            del self.image_cache[key]

    def _fetch_openai_models(self) -> List[Dict[str, Any]]:
        """Fetch available models from the OpenAI API"""
        current_time = time.time()

        # Return cached models if still valid
        if (
            current_time - self.model_list_cache["timestamp"]
            < self.MODEL_LIST_CACHE_EXPIRATION
            and self.model_list_cache["models"]
        ):
            return self.model_list_cache["models"]

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
                models = data.get("data", [])

                # Update cache
                self.model_list_cache = {"models": models, "timestamp": current_time}

                return models
            else:
                logging.error(
                    f"Failed to fetch models: HTTP {response.status_code}: {response.text}"
                )
                # Return empty list or cached models if available
                return self.model_list_cache["models"]
        except Exception as e:
            logging.error(f"Error fetching OpenAI models: {str(e)}")
            return self.model_list_cache["models"]

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

            self.clean_expired_cache()

            image_url = image_data.get("image_url", {}).get("url", "")
            image_key = image_url
            if image_url.startswith("data:image"):
                image_key = image_url.split(",", 1)[1] if "," in image_url else ""

            if image_key in self.image_cache:
                logging.info(f"Using cached image description for {image_key[:30]}...")
                return self.image_cache[image_key].description

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

            self.image_cache[image_key] = CacheEntry(description)

            if len(self.image_cache) > 100:
                oldest_key = min(
                    self.image_cache.keys(), key=lambda k: self.image_cache[k].timestamp
                )
                del self.image_cache[oldest_key]

            return description

        except Exception as e:
            logging.error(f"Error processing image with Gemini: {str(e)}")
            return f"[Error processing image: {str(e)}]"
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

        for message in messages:
            images, text = self.extract_images_and_text(message)

            if images:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Found {len(images)} image(s) to process",
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
                processed_messages.append(message)

        return processed_messages

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
                return self._stream_response(
                    url=self.valves.OPENAI_API_URL,
                    headers=headers,
                    payload=payload,
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
        self, url: str, headers: dict, payload: dict, __event_emitter__=None
    ) -> AsyncIterator[str]:
        try:
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
