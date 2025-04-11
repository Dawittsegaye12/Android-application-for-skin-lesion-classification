import numpy as np

class ModelConfig:
    # Path to the model file
    MODEL_FILENAME = "skin_analyzer.tflite"

    # Input image size expected by the model
    INPUT_IMG_SIZE_WIDTH = 450
    INPUT_IMG_SIZE_HEIGHT = 450
    
    # Model input size calculation (for TensorFlow Lite models)
    FLOAT_TYPE_SIZE = 4  # Size of a float in bytes
    PIXEL_SIZE = 3  # RGB channels
    MODEL_INPUT_SIZE = FLOAT_TYPE_SIZE * INPUT_IMG_SIZE_WIDTH * INPUT_IMG_SIZE_HEIGHT * PIXEL_SIZE

    # Output labels corresponding to the model's predictions
    OUTPUT_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

    # Number of classification results to display
    MAX_CLASSIFICATION_RESULTS = 7
    
    # Minimum classification confidence threshold
    CLASSIFICATION_THRESHOLD = 0.1

    @staticmethod
    def get_model_input_size():
        """Returns the model input size (in bytes) for a single image."""
        return ModelConfig.MODEL_INPUT_SIZE

    @staticmethod
    def get_output_labels():
        """Returns the list of output labels."""
        return ModelConfig.OUTPUT_LABELS

    @staticmethod
    def get_classification_threshold():
        """Returns the classification threshold."""
        return ModelConfig.CLASSIFICATION_THRESHOLD

    @staticmethod
    def get_max_classification_results():
        """Returns the maximum number of classification results to display."""
        return ModelConfig.MAX_CLASSIFICATION_RESULTS


# Example usage
if __name__ == "__main__":
    print(f"Model filename: {ModelConfig.MODEL_FILENAME}")
    print(f"Input image size: {ModelConfig.INPUT_IMG_SIZE_WIDTH}x{ModelConfig.INPUT_IMG_SIZE_HEIGHT}")
    print(f"Model input size (bytes): {ModelConfig.get_model_input_size()}")
    print(f"Output labels: {ModelConfig.get_output_labels()}")
    print(f"Classification threshold: {ModelConfig.get_classification_threshold()}")
    print(f"Max classification results: {ModelConfig.get_max_classification_results()}")
