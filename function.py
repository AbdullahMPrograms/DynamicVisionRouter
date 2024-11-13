from pydantic import BaseModel, Field
from typing import Callable, Awaitable, Any, Optional, Literal
import json
import copy

from open_webui.utils.misc import get_last_user_message_item


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority level for the filter operations.",
        )
        vision_model_id: str = Field(
            default="",
            description="The identifier of the vision model to be used for processing images. Note: Compatibility is provider-specific; ollama models can only route to ollama models, and OpenAI models to OpenAI models respectively.",
        )
        skip_reroute_models: list[str] = Field(
            default_factory=list,
            description="A list of model identifiers that should not be re-routed to the chosen vision model.",
        )
        enabled_for_admins: bool = Field(
            default=True,
            description="Whether dynamic vision routing is enabled for admin users.",
        )
        enabled_for_users: bool = Field(
            default=True,
            description="Whether dynamic vision routing is enabled for regular users.",
        )
        status: bool = Field(
            default=False,
            description="A flag to enable or disable the status indicator. Set to True to enable status updates.",
        )
        vision_prompt: str = Field(
            default="",
            description="The prompt to be sent to the vision model when processing images.",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        self._vision_output = None
        self._original_message = None
        pass

    def _create_vision_messages(self) -> list:
        """Create messages for vision model with only the image and prompt."""
        return (
            [
                {
                    "role": "user",
                    "content": self.valves.vision_prompt,
                    "images": self._original_message.get("images"),
                }
            ]
            if "images" in self._original_message
            else [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": self.valves.vision_prompt}]
                    + [
                        item
                        for item in self._original_message.get("content", [])
                        if item.get("type") == "image_url"
                    ],
                }
            ]
        )

    def _create_main_model_messages(self, messages: list) -> list:
        """Create messages for main model including vision output."""
        # Get all messages except the last one (which has the images)
        previous_messages = messages[:-1]

        # Add vision model's output as assistant message
        vision_response = {"role": "assistant", "content": self._vision_output}

        # Add the original user message without images
        user_message = copy.deepcopy(self._original_message)
        if "images" in user_message:
            del user_message["images"]
            if isinstance(user_message.get("content"), str):
                # If the original message had text content, preserve it
                if user_message["content"].strip():
                    user_message["content"] = user_message["content"]
                else:
                    user_message["content"] = ""
        elif isinstance(user_message.get("content"), list):
            user_message["content"] = [
                item
                for item in user_message["content"]
                if item.get("type") != "image_url"
            ]
            if not user_message["content"]:
                user_message["content"] = ""

        return previous_messages + [vision_response, user_message]

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __model__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        if __model__["id"] in self.valves.skip_reroute_models:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        user_message = get_last_user_message_item(messages)
        if user_message is None:
            return body

        has_images = user_message.get("images") is not None
        if not has_images:
            user_message_content = user_message.get("content")
            if user_message_content is not None and isinstance(
                user_message_content, list
            ):
                has_images = any(
                    item.get("type") == "image_url" for item in user_message_content
                )

        if has_images:
            if self.valves.vision_model_id:
                if __model__["id"] == self.valves.vision_model_id:
                    # Store vision model's response and return empty response
                    self._vision_output = (
                        body.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    return {"choices": [{"message": {"content": ""}}]}
                else:
                    # Main model processing
                    if self._vision_output is None:
                        try:
                            if self.valves.status:
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": f"Routing to vision model {self.valves.vision_model_id}",
                                            "done": False,
                                        },
                                    }
                                )

                            # Store original message and create vision request
                            self._original_message = copy.deepcopy(user_message)
                            vision_body = copy.deepcopy(body)
                            vision_body["model"] = self.valves.vision_model_id
                            vision_body["messages"] = self._create_vision_messages()

                            return vision_body
                        except Exception as e:
                            if self.valves.status:
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": f"Error in vision processing: {str(e)}",
                                            "done": True,
                                        },
                                    }
                                )
                            raise e
                    else:
                        try:
                            if self.valves.status:
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": "Processing with main model",
                                            "done": True,
                                        },
                                    }
                                )

                            # Create messages for main model
                            body["messages"] = self._create_main_model_messages(
                                messages
                            )

                            # Clear stored data
                            self._vision_output = None
                            self._original_message = None

                            return body

                        except Exception as e:
                            if self.valves.status:
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": f"Error processing with main model: {str(e)}",
                                            "done": True,
                                        },
                                    }
                                )
                            raise e
            else:
                if self.valves.status:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "No vision model ID provided, routing could not be completed.",
                                "done": True,
                            },
                        }
                    )
        return body
