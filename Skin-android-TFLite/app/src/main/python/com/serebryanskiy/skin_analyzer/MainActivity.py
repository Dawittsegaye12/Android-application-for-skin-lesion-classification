import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

class MainActivity:
    def __init__(self, root):
        self.root = root
        self.root.title("Skin Analyzer")
        
        # Load the classifier (assuming it is a TensorFlow Lite model)
        try:
            self.classifier = Classifier.load_model("path_to_model.tflite")
        except Exception as e:
            messagebox.showerror("Error", "Model couldn't be loaded. Check logs for details.")
            print(str(e))
        
        # Set up the UI components
        self.vCamera = cv2.VideoCapture(0)  # Capture from the webcam
        
        self.ivPreview = tk.Label(root)
        self.ivPreview.pack()

        self.ivFinalPreview = tk.Label(root)
        self.ivFinalPreview.pack()

        self.tvClassification = tk.Label(root, text="Classification result will appear here")
        self.tvClassification.pack()

        self.btnTakePhoto = tk.Button(root, text="Take Photo", command=self.on_take_photo)
        self.btnTakePhoto.pack()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_camera_feed()

    def on_take_photo(self):
        """Capture a photo from the camera and process it."""
        ret, frame = self.vCamera.read()
        if ret:
            # Convert to grayscale for preview
            square_bitmap = cv2.resize(frame, (self.get_screen_width(), self.get_screen_width()))
            preview_image = Image.fromarray(cv2.cvtColor(square_bitmap, cv2.COLOR_BGR2RGB))
            preview_image.thumbnail((250, 250))  # Resize for preview display
            self.ivPreview.img = ImageTk.PhotoImage(preview_image)
            self.ivPreview.config(image=self.ivPreview.img)

            # Process image for classification
            preprocessed_image = ImageUtils.prepare_image_for_classification(preview_image)
            self.ivFinalPreview.img = ImageTk.PhotoImage(preprocessed_image)
            self.ivFinalPreview.config(image=self.ivFinalPreview.img)

            # Make predictions
            recognitions = self.classifier.recognize_image(preprocessed_image)
            self.tvClassification.config(text=str(recognitions))

    def update_camera_feed(self):
        """Update the camera feed on the screen."""
        ret, frame = self.vCamera.read()
        if ret:
            # Show the camera frame in the preview window
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image.thumbnail((250, 250))
            self.ivPreview.img = ImageTk.PhotoImage(image)
            self.ivPreview.config(image=self.ivPreview.img)

        self.root.after(10, self.update_camera_feed)

    def get_screen_width(self):
        """Get the width of the screen."""
        return self.root.winfo_screenwidth()

    def on_close(self):
        """Close the camera feed and the application."""
        self.vCamera.release()
        self.root.destroy()


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


class ModelConfig:
    # These values should match the values used in your original Java code.
    CLASSIFICATION_THRESHOLD = 0.5
    MAX_CLASSIFICATION_RESULTS = 3
    OUTPUT_LABELS = ["Label 1", "Label 2", "Label 3"]  # Example labels, adjust as needed.
    INPUT_IMG_SIZE_WIDTH = 224
    INPUT_IMG_SIZE_HEIGHT = 224


# Simulating a classifier class as an example
class Classifier:
    @staticmethod
    def load_model(model_path):
        # Load your TensorFlow Lite model here and return an instance
        return Classifier()

    def recognize_image(self, image: Image.Image):
        # Simulate some result from the image recognition model
        return [("Label 1", 0.95), ("Label 2", 0.80), ("Label 3", 0.60)]


# Create the main window (root)
root = tk.Tk()
app = MainActivity(root)

# Run the application
root.mainloop()
