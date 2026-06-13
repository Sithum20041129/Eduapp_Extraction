import io
import logging
import threading

import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image as VertexImage
from PIL import Image
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

import config

logger = logging.getLogger("ai_service.model_loader")

# Retry only on transient upstream errors (429/503/timeouts), not on bugs.
try:
    from google.api_core import exceptions as _gexc
    _RETRYABLE = (
        _gexc.ResourceExhausted,
        _gexc.ServiceUnavailable,
        _gexc.DeadlineExceeded,
        _gexc.InternalServerError,
        _gexc.TooManyRequests,
    )
except Exception:  # pragma: no cover - google libs always present at runtime
    _RETRYABLE = (Exception,)


@retry(
    reraise=True,
    stop=stop_after_attempt(config.MODEL_MAX_RETRIES),
    wait=wait_random_exponential(multiplier=1, max=20),
    retry=retry_if_exception_type(_RETRYABLE),
)
def _generate_with_retry(model, inputs):
    """Call generate_content with exponential backoff on transient errors."""
    return model.generate_content(inputs)


class ModelLoader:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._model = None
        self._model_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        # Double-checked locking so concurrent first-requests don't each
        # construct an instance / re-initialise Vertex AI.
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = ModelLoader()
        return cls._instance

    def load_model(self):
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:  # another thread won the race
                return
            logger.info("Configuring Vertex AI...")
            try:
                if not config.GCP_PROJECT_ID:
                    raise ValueError("GCP_PROJECT_ID environment variable is not set")

                vertexai.init(project=config.GCP_PROJECT_ID, location=config.GCP_LOCATION)
                model_name = config.PRIMARY_MODEL
                self._model = GenerativeModel(model_name)
                logger.info(
                    "Vertex AI model '%s' configured (project=%s location=%s)",
                    model_name, config.GCP_PROJECT_ID, config.GCP_LOCATION,
                )
            except Exception as e:
                logger.error("Error configuring Vertex AI: %s", e)
                logger.info("Attempting fallback to '%s'...", config.FALLBACK_MODEL)
                try:
                    self._model = GenerativeModel(config.FALLBACK_MODEL)
                    logger.info("Fallback to %s successful.", config.FALLBACK_MODEL)
                except Exception as ex:
                    logger.error("Fallback failed: %s", ex)
                    raise

    def get_model(self):
        if self._model is None:
            self.load_model()
        return self._model

    def predict(self, image_input, prompt_text="Extract all text from this image."):
        model = self.get_model()
        
        # Prepare inputs
        inputs = [prompt_text]
        image_included = False
        
        if isinstance(image_input, Image.Image):
             # Simple heuristic: if it's the dummy 10x10 white image, skip it
            if image_input.size == (10, 10):
                # It's likely the dummy for text-only marking
                pass
            else:
                # Convert PIL Image to Vertex AI Part
                img_byte_arr = io.BytesIO()
                image_input.save(img_byte_arr, format=image_input.format or 'PNG')
                img_bytes = img_byte_arr.getvalue()
                image_part = Part.from_data(data=img_bytes, mime_type="image/png")
                inputs.append(image_part)
                image_included = True
        
        try:
            response = _generate_with_retry(model, inputs)

            # Extract usage metadata from response
            usage_metadata = {
                'prompt_token_count': getattr(response.usage_metadata, 'prompt_token_count', 0),
                'candidates_token_count': getattr(response.usage_metadata, 'candidates_token_count', 0),
                'total_token_count': getattr(response.usage_metadata, 'total_token_count', 0),
                'image_included': image_included
            }

            return response.text, usage_metadata
        except Exception as e:
            # Handle potential API errors (after retries are exhausted)
            logger.error("generate_content failed after retries: %s", e)
            error_metadata = {
                'prompt_token_count': 0,
                'candidates_token_count': 0,
                'total_token_count': 0,
                'image_included': image_included
            }
            return f"Error generating content: {e}", error_metadata

    def predict_with_parts(self, parts: list, prompt_text: str):
        """
        Send arbitrary Part objects (e.g. PDFs) + prompt to the model.
        
        Args:
            parts: List of vertexai Part objects (PDF bytes, images, etc.)
            prompt_text: The prompt to send with the parts
            
        Returns:
            Tuple of (response_text, usage_metadata)
        """
        model = self.get_model()
        inputs = [prompt_text] + parts

        try:
            response = _generate_with_retry(model, inputs)
            usage_metadata = {
                'prompt_token_count': getattr(response.usage_metadata, 'prompt_token_count', 0),
                'candidates_token_count': getattr(response.usage_metadata, 'candidates_token_count', 0),
                'total_token_count': getattr(response.usage_metadata, 'total_token_count', 0),
                'image_included': True
            }
            # Extract text from all parts (response.text fails with multi-part responses)
            try:
                text = response.text
            except Exception:
                # Manually concatenate text from all candidate parts
                text = ""
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        text += part.text
            return text, usage_metadata
        except Exception as e:
            logger.error("generate_content (parts) failed after retries: %s", e)
            error_metadata = {
                'prompt_token_count': 0,
                'candidates_token_count': 0,
                'total_token_count': 0,
                'image_included': True
            }
            return f"Error generating content: {e}", error_metadata
