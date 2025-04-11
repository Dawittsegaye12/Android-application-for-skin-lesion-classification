from PIL import Image

class ImageUtils:
    
    @staticmethod
    def prepare_image_for_classification(image: Image.Image) -> Image.Image:
        """Prepare image for classification by resizing it to the required input size."""
        # Resize the image to the specified input size for the model
        resized_image = image.resize(
            (ModelConfig.INPUT_IMG_SIZE_WIDTH, ModelConfig.INPUT_IMG_SIZE_HEIGHT),
            Image.ANTIALIAS  # or Image.LANCZOS for high-quality resampling
        )
        return resized_image
