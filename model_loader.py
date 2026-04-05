import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image as VertexImage
from PIL import Image
import os
import io

from dotenv import load_dotenv

load_dotenv()

class ModelLoader:
    _instance = None
    _model = None
    _project_id = os.getenv("GCP_PROJECT_ID")
    _location = os.getenv("GCP_LOCATION", "asia-south1")
    _credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelLoader()
        return cls._instance

    def load_model(self):
        if self._model is None:
            print("Configuring Vertex AI...")
            try:
                if not self._project_id:
                    raise ValueError("GCP_PROJECT_ID environment variable is not set")
                
                # Initialize Vertex AI with project and location
                vertexai.init(
                    project=self._project_id,
                    location=self._location
                )

                # Using the specific model requested.
                model_name = "gemini-2.5-flash" 

                self._model = GenerativeModel(model_name)
                print(f"Vertex AI model '{model_name}' configured successfully.")
                print(f"  Project: {self._project_id}")
                print(f"  Location: {self._location}")
            except Exception as e:
                print(f"Error configuring Vertex AI: {e}")
                # Fallback purely for robustness
                print("Attempting fallback to 'gemini-1.5-flash'...")
                try:
                    self._model = GenerativeModel("gemini-1.5-flash")
                    print("Fallback to gemini-1.5-flash successful.")
                except Exception as ex:
                    print(f"Fallback failed: {ex}")
                    raise ex

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
            response = model.generate_content(inputs)
            
            # Extract usage metadata from response
            usage_metadata = {
                'prompt_token_count': getattr(response.usage_metadata, 'prompt_token_count', 0),
                'candidates_token_count': getattr(response.usage_metadata, 'candidates_token_count', 0),
                'total_token_count': getattr(response.usage_metadata, 'total_token_count', 0),
                'image_included': image_included
            }
            
            return response.text, usage_metadata
        except Exception as e:
            # Handle potential API errors
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
            response = model.generate_content(inputs)
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
            error_metadata = {
                'prompt_token_count': 0,
                'candidates_token_count': 0,
                'total_token_count': 0,
                'image_included': True
            }
            return f"Error generating content: {e}", error_metadata
