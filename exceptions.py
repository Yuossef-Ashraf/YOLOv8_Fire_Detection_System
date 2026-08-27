"""
Custom Exception Classes for YOLOv8 Fire & Smoke Detection System.
"""


class FireDetectionError(Exception):
    """Base exception for all Fire Detection System errors."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }


class ModelNotFoundError(FireDetectionError):
    """Raised when YOLO weights file (.pt) is not found."""
    pass


class ModelInferenceError(FireDetectionError):
    """Raised when model forward pass or prediction fails."""
    pass


class VideoStreamError(FireDetectionError):
    """Raised when camera stream or video file cannot be opened or read."""
    pass


class HardwareAlarmError(FireDetectionError):
    """Raised when serial communication with hardware alarm (Arduino) fails."""
    pass
