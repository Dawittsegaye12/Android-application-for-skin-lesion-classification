import logging

class Classification:
    def __init__(self, title: str, confidence: float):
        # Initialize title and confidence
        self.title = title
        self.confidence = confidence
        self._validate()

    def _validate(self):
        """Ensure that the confidence value is within the valid range."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0 and 1.")

    def __str__(self):
        """Return the string representation of the classification."""
        return f"{self.title} ({self.confidence * 100:.1f}%)"

    def to_dict(self):
        """Return the classification data as a dictionary."""
        return {"title": self.title, "confidence": self.confidence}

    def log_classification(self):
        """Log the classification information for tracking."""
        logging.basicConfig(level=logging.INFO)
        logging.info(f"Classification: {self.title} with confidence: {self.confidence * 100:.1f}%")

    def update_confidence(self, new_confidence: float):
        """Update the confidence score if the new value is valid."""
        if not (0.0 <= new_confidence <= 1.0):
            raise ValueError("Confidence must be between 0 and 1.")
        self.confidence = new_confidence
        logging.info(f"Updated confidence to {new_confidence * 100:.1f}% for {self.title}.")
