import os
import json
import logging
import requests
import aiohttp
import google.generativeai as genai  # type: ignore[import-not-found]
from typing import (
    List,
    Dict,
    Optional,
    AsyncIterator,
    Tuple,
    Any,
)
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message  # type: ignore[import-not-found]


class ModelConfig(BaseModel):
    name: str
    alias: Optional[str] = None

class Pipe:
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
            default=os.getenv("GEMINI_SECTION_TITLE", "Image Descriptions"),
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
            # 2s connect timeout, 5s read timeout, needed for latency reasons
            response = requests.get(models_url, headers=headers, timeout=(2, 5))
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
        models_to_use = []
        if self.valves.OPENAI_MODELS.lower() == "all":
            all_models = self._fetch_openai_models()
            for model in all_models:
                models_to_use.append(
                    {
                        "id": f"{self.id}/{model.get('id')}",
                        "name": model.get("id"),
                        "supports_vision": True,
                    }
                )
        else:
            model_names = [n.strip() for n in self.valves.OPENAI_MODELS.split(",")]
            for name in model_names:
                if name:
                    models_to_use.append(
                        {
                            "id": f"{self.id}/{name}",
                            "name": name,
                            "supports_vision": True,
                        }
                    )
        return models_to_use

    def pipes(self) -> List[dict]:
        return self.get_available_models()

    def extract_images_and_text(self, message: Dict) -> Tuple[List[Dict], str]:
        # Extract images and text from a message content, separating for processing
        images, text_parts = [], []
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
        """Process an image with Gemini and return its description."""
        processing_message = None
        try:
            if not self.valves.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is required for image processing")

            image_url = image_data.get("image_url", {}).get("url", "")

            genai.configure(api_key=self.valves.GOOGLE_API_KEY)
            model = genai.GenerativeModel(self.valves.GEMINI_MODEL_NAME)

            if image_url.startswith("data:image"):
                base64_data = image_url.split(",", 1)[1] if "," in image_url else ""
                image_part = {
                    "inline_data": {"mime_type": "image/jpeg", "data": base64_data}
                }
            else:
                image_part = {"image_url": image_url}

            # Use the configurable prompt from valves
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
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Generating Response...",
                            "done": False,
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
        logging.debug("Processing %d messages for image content", len(messages))

        # Find the last user message with images (that's the only one we'll process with Gemini)
        last_user_msg_with_images_idx = -1
        for i in range(len(messages) - 1, -1, -1):  # Iterate backward
            if messages[i].get("role") == "user":
                images, _ = self.extract_images_and_text(messages[i])
                if images:
                    last_user_msg_with_images_idx = i
                    logging.debug("Found user message with %d images at index %d", len(images), i)
                    break

        # Process all messages (ensure all messages are converted to text format)
        for i, message in enumerate(messages):
            images, text = self.extract_images_and_text(message)

            if images:
                if i == last_user_msg_with_images_idx:
                    logging.info("Processing %d images in latest user message", len(images))

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
                    logging.debug("Skipping image processing for older message (index %d, %d images)", i, len(images))
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

    def format_usage_status(self, usage: Dict[str, Any], timings: Optional[Dict[str, Any]] = None) -> str:
        """Return single-line usage summary: "TG: 81.75 T/s | PP: 70.61 T/s | PT: 74 | GT: 150 tokens | TT: 224 tokens | 1.83 sec"""
        try:
            # Get token counts from usage
            completion_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
            
            # Get timing/throughput data from timings if available
            if timings:
                predicted_per_second = timings.get("predicted_per_second", 0.0)
                prompt_per_second = timings.get("prompt_per_second", 0.0)
                predicted_ms = timings.get("predicted_ms", 0.0)
                prompt_ms = timings.get("prompt_ms", 0.0)
                prompt_n = timings.get("prompt_n", prompt_tokens)
                predicted_n = timings.get("predicted_n", completion_tokens)
                total_ms = prompt_ms + predicted_ms
            else:
                predicted_per_second = 0.0
                prompt_per_second = 0.0
                prompt_n = prompt_tokens
                total_ms = 0.0
                predicted_n = completion_tokens
            
            # Format the components
            if predicted_per_second > 0:
                tg_str = f"{predicted_per_second:.2f} T/s"
                
            if prompt_per_second > 0:
                pp_str = f"PP: {prompt_per_second:.2f} T/s"
                
            if total_ms > 0:
                time_str = f"{total_ms/1000.0:.2f} sec"
            
            result = f"{tg_str} | {completion_tokens} tokens | {pp_str} | PT: {prompt_n} tokens | TT: {total_tokens} tokens | {time_str}"
            return result
        except Exception as e:
            logging.error("Error formatting usage status: %s", str(e))
            return "Error formatting usage status"

    def build_dynamic_payload(self, body: Dict, model: str, messages: List[Dict]) -> Dict[str, Any]:
        """
        Dynamically build the payload by extracting ALL parameters from the request body.
        This automatically supports any new or custom parameters without requiring code changes.
        """
        excluded_parameters = {
            'model',    
            'messages',  
        }
        
        # Start with required parameters
        payload = {
            "model": model,
            "messages": messages
        }
        
        # Add ALL parameters from the body except excluded ones and filter out None values
        for param, value in body.items():
            if param not in excluded_parameters and value is not None:
                payload[param] = value
        
        logging.debug("Built dynamic payload with parameters: %s", list(payload.keys()))
        return payload

    async def pipe(
        self, body: Dict, __event_emitter__=None
    ) -> AsyncIterator[str]:
        if not all([self.valves.OPENAI_API_URL, self.valves.OPENAI_API_KEY]):
            error_msg = "Error: OPENAI_API_URL and OPENAI_API_KEY are required"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            yield error_msg
            return

        try:
            requested_model = body.get("model", "").split("/")[-1]

            system_message, messages = pop_system_message(body.get("messages", []))
            processed_messages = await self.process_messages(
                messages, __event_emitter__
            )
            if system_message:
                processed_messages.insert(0, system_message)

            payload = self.build_dynamic_payload(body, requested_model, processed_messages)

            logging.info(
                "Prepared payload: model=%s, max_tokens=%s",
                payload.get("model"),
                payload.get("max_tokens"),
            )

            headers = {
                "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }

            gemini_desc = ""
            if (
                self.valves.SHOW_GEMINI_DESCRIPTIONS
                and self.current_gemini_descriptions
            ):
                gemini_desc = self.format_gemini_descriptions()
            async for chunk in self._stream_response(
                url=self.valves.OPENAI_API_URL,
                headers=headers,
                payload=payload,
                gemini_descriptions=gemini_desc,
                __event_emitter__=__event_emitter__,
            ):
                yield chunk

        except Exception as e:
            err = f"Error: {str(e)}"
            logging.exception("Unhandled error in pipe: %s", err)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": err, "done": True}}
                )
            yield err
            return

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
                logging.debug("Yielding Gemini descriptions (%d chars)", len(gemini_descriptions))
                yield gemini_descriptions

            # Create a client session with no timeout for streaming
            timeout = aiohttp.ClientTimeout(total=None)  # Disable timeout completely
            async with aiohttp.ClientSession(timeout=timeout) as session:
                logging.info("Starting streaming request to %s", url)
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_msg = (
                            f"Error: HTTP {response.status}: {await response.text()}"
                        )
                        logging.error("Streaming error: %s", error_msg)
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {"description": error_msg, "done": True},
                                }
                            )
                        yield error_msg
                        return

                    usage_captured: Optional[Dict[str, Any]] = None
                    timings_captured: Optional[Dict[str, Any]] = None
                    async for line in response.content:
                        if line and line.startswith(b"data: "):
                            try:
                                line_text = line[6:].decode("utf-8")
                                if line_text.strip() == "[DONE]":
                                    # On completion, push usage as a status if available; otherwise emit generic completion
                                    logging.debug("Streaming completed")
                                    if __event_emitter__:
                                        if usage_captured:
                                            stats = self.format_usage_status(usage_captured, timings_captured)
                                            await __event_emitter__({
                                                "type": "status", 
                                                "data": {"description": stats, "done": True}
                                            })
                                        else:
                                            await __event_emitter__({
                                                "type": "status",
                                                "data": {"description": "Request completed", "done": True},
                                            })
                                    break

                                data = json.loads(line_text)
                                # If usage information is included in the stream, capture it
                                if isinstance(data, dict) and data.get("usage"):
                                    usage_captured = data["usage"]
                                    # Also capture timings if present
                                    if "timings" in data:
                                        timings_captured = data["timings"]
                                    # Do not yield anything for usage-only messages
                                    if not (
                                        "choices" in data
                                        and len(data["choices"]) > 0
                                        and "delta" in data["choices"][0]
                                    ):
                                        continue
                                if (
                                    "choices" in data
                                    and len(data["choices"]) > 0
                                    and "delta" in data["choices"][0]
                                ):
                                    delta_content = data["choices"][0]["delta"].get(
                                        "content"
                                    )
                                    # If delta_content is None, yield an empty string. Otherwise, yield the content.
                                    yield (
                                        delta_content
                                        if delta_content is not None
                                        else ""
                                    )
                            except json.JSONDecodeError as e:
                                logging.error(
                                    f"Failed to parse streaming response: {e}"
                                )
                                continue

        except Exception as e:
            error_msg = f"Stream error: {str(e)}"
            logging.exception("Unhandled streaming error: %s", error_msg)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            yield error_msg
