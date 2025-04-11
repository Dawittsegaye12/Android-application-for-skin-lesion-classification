import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
import tensorflow.lite as tflite

class Classifier:
    def __init__(self, interpreter):
        self.interpreter = interpreter

    @staticmethod
    def load_model(model_path):
        """Load TensorFlow Lite model."""
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return Classifier(interpreter)

    def recognize_image(self, image: Image.Image):
        """Recognize image and return sorted classification results."""
        byte_buffer = self.convert_image_to_byte_buffer(image)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        self.interpreter.set_tensor(input_details[0]['index'], byte_buffer)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(output_details[0]['index'])
        return self.get_sorted_result(output)

    def convert_image_to_byte_buffer(self, image: Image.Image):
        """Convert image to a byte buffer compatible with the model input."""
        image = image.resize((ModelConfig.INPUT_IMG_SIZE_WIDTH, ModelConfig.INPUT_IMG_SIZE_HEIGHT))
        image = np.array(image).astype(np.float32)
        
        # Normalize pixel values to [0, 1] range
        image = np.mean(image, axis=-1) / 255.0  # Convert to grayscale and normalize

        # Flatten image to a byte buffer
        byte_buffer = np.reshape(image, (1, ModelConfig.INPUT_IMG_SIZE_WIDTH * ModelConfig.INPUT_IMG_SIZE_HEIGHT)).astype(np.float32)
        return byte_buffer

    def get_sorted_result(self, output):
        """Get sorted classification results."""
        results = []
        for i in range(len(ModelConfig.OUTPUT_LABELS)):
            confidence = output[0][i]
            if confidence > ModelConfig.CLASSIFICATION_THRESHOLD:
                results.append((ModelConfig.OUTPUT_LABELS[i], confidence))

        # Sort by confidence
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        return [Classification(label, conf) for label, conf in sorted_results]

class Classification:
    def __init__(self, title: str, confidence: float):
        self.title = title
        self.confidence = confidence

    def __str__(self):
        return f"{self.title} ({self.confidence * 100:.1f}%)"


class ModelConfig:
    # These values should match the values used in your original Java code.
    CLASSIFICATION_THRESHOLD = 0.5
    MAX_CLASSIFICATION_RESULTS = 3
    OUTPUT_LABELS = ["Label 1", "Label 2", "Label 3"]  # Example labels, adjust as needed.
    INPUT_IMG_SIZE_WIDTH = 224
    INPUT_IMG_SIZE_HEIGHT = 224
