"""
YOLOv8 Fire and Smoke Detection Engine.
Provides modular inference, frame annotation, metrics tracking, and hardware alarm interfacing.
"""

import os
import sys
import time
import math
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path

from exceptions import (
    ModelNotFoundError,
    ModelInferenceError,
    VideoStreamError,
    HardwareAlarmError
)
from logging_config import setup_logging

logger = logging.getLogger("fire_detection.detector")


class HardwareAlarmController:
    """
    Optional serial communication controller for Arduino hardware buzzers & relays.
    """
    def __init__(self, port: Optional[str] = None, baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self._is_active = False
        self._initialize_connection()

    def _initialize_connection(self):
        if not self.port:
            return
        try:
            import serial
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            self._is_active = True
            logger.info(f"Hardware Alarm Controller connected to {self.port} at {self.baudrate} baud.")
        except Exception as e:
            logger.warning(f"Could not connect to Hardware Alarm on {self.port}: {e}. Operating in software-only mode.")
            self._is_active = False

    def trigger_alarm(self, state: bool):
        """Send alarm trigger byte (1 = active, 0 = clear) to Arduino."""
        if not self._is_active or not self.serial_conn:
            return
        try:
            byte_cmd = b'1' if state else b'0'
            self.serial_conn.write(byte_cmd)
        except Exception as e:
            logger.error(f"Hardware communication error: {e}")

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.trigger_alarm(False)
                self.serial_conn.close()
            except Exception:
                pass


class FireSmokeDetector:
    """
    Real-time Fire & Smoke detection using YOLOv8 neural network.
    """
    def __init__(
        self,
        weights_path: str = "fire.pt",
        conf_threshold: float = 0.45,
        device: str = "auto"
    ):
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = None
        self.class_names = {0: "Fire", 1: "Smoke", 2: "Neutral"}
        self._load_model()

    def _load_model(self):
        """Load YOLOv8 weights file."""
        if not self.weights_path.exists():
            raise ModelNotFoundError(f"Model weights file not found at: {self.weights_path.resolve()}")

        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.weights_path))
            
            # Extract class names if defined in weights
            if hasattr(self.model, "names") and self.model.names:
                self.class_names = self.model.names

            logger.info(f"YOLOv8 Fire Detection model loaded successfully from {self.weights_path}. Classes: {self.class_names}")
        except ImportError:
            logger.warning("ultralytics package not installed. Inference methods will require ultralytics.")
        except Exception as e:
            raise ModelInferenceError(f"Failed to load YOLO model: {e}")

    def predict_frame(self, frame: Any) -> List[Dict[str, Any]]:
        """
        Run forward pass on a single frame / image array.

        Returns:
            List of detection dicts: [{"bbox": (x1, y1, x2, y2), "conf": float, "class_id": int, "label": str}]
        """
        if self.model is None:
            raise ModelInferenceError("Model is not initialized.")

        try:
            results = self.model(frame, verbose=False, device=None if self.device == "auto" else self.device)
            detections = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf[0].item() if hasattr(box.conf[0], 'item') else box.conf[0])
                    if conf >= self.conf_threshold:
                        xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], 'tolist') else box.xyxy[0]
                        x1, y1, x2, y2 = [int(v) for v in xyxy]
                        cls_id = int(box.cls[0].item() if hasattr(box.cls[0], 'item') else box.cls[0])
                        label = self.class_names.get(cls_id, f"Class_{cls_id}")

                        detections.append({
                            "bbox": (x1, y1, x2, y2),
                            "confidence": round(conf, 3),
                            "class_id": cls_id,
                            "label": label
                        })

            return detections
        except Exception as e:
            logger.error(f"Inference error on frame: {e}")
            raise ModelInferenceError(f"Prediction failed: {e}")

    def annotate_frame(
        self,
        frame: Any,
        detections: List[Dict[str, Any]],
        show_fps: Optional[float] = None
    ) -> Any:
        """
        Draw clean bounding boxes, labels, and status badges on the frame.
        """
        try:
            import cv2
        except ImportError:
            return frame

        annotated = frame.copy()
        
        # Color definitions (BGR)
        color_fire = (0, 30, 230)     # Bright Crimson Red
        color_smoke = (40, 140, 230)  # Dark Orange / Amber
        color_default = (0, 255, 0)   # Green

        fire_detected = False

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            
            # Select color based on class
            if "fire" in label.lower():
                color = color_fire
                fire_detected = True
            elif "smoke" in label.lower():
                color = color_smoke
            else:
                color = color_default

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw tag background
            tag_text = f"{label} {int(conf * 100)}%"
            (w, h), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - 25)), (x1 + w + 10, y1), color, -1)
            cv2.putText(
                annotated,
                tag_text,
                (x1 + 5, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # Draw Global HUD / Status
        hud_bg_color = (0, 0, 180) if fire_detected else (30, 30, 30)
        hud_text = "[ALERT] FIRE DETECTED!" if fire_detected else "[NORMAL] System Monitoring"
        cv2.rectangle(annotated, (10, 10), (320, 45), hud_bg_color, -1)
        cv2.putText(annotated, hud_text, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if show_fps is not None:
            fps_text = f"FPS: {show_fps:.1f}"
            cv2.putText(annotated, fps_text, (annotated.shape[1] - 120, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return annotated
