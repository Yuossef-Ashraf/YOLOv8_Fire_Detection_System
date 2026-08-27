"""
Command-Line Interface for YOLOv8 Fire & Smoke Detection System.
Supports inference on images, pre-recorded video files, and real-time webcam feeds.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

from detector import FireSmokeDetector, HardwareAlarmController
from exceptions import VideoStreamError, ModelNotFoundError
from logging_config import setup_logging

logger = logging.getLogger("fire_detection.cli")


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Real-Time Fire & Smoke Detection CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: '0' for default webcam, path to video file (.mp4, .avi), or image file (.jpg, .png)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="fire.pt",
        help="Path to YOLOv8 trained weights file"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Confidence detection threshold (0.0 - 1.0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "0"],
        help="Inference device"
    )
    parser.add_argument(
        "--view",
        action="store_true",
        default=True,
        help="Display output in OpenCV graphical window"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save annotated output (image or video)"
    )
    parser.add_argument(
        "--arduino-port",
        type=str,
        default=None,
        help="COM port for Arduino serial hardware alarm (e.g. COM6 or /dev/ttyUSB0)"
    )
    return parser.parse_args()


def process_image(detector: FireSmokeDetector, image_path: str, save_path: str = None, view: bool = True):
    import cv2
    frame = cv2.imread(image_path)
    if frame is None:
        raise VideoStreamError(f"Could not load image from: {image_path}")

    detections = detector.predict_frame(frame)
    annotated = detector.annotate_frame(frame, detections)

    logger.info(f"Detections in {image_path}: {len(detections)} targets found.")
    for d in detections:
        logger.info(f" -> {d['label']} ({int(d['confidence']*100)}%) at {d['bbox']}")

    if save_path:
        cv2.imwrite(save_path, annotated)
        logger.info(f"Annotated image saved to: {save_path}")

    if view:
        cv2.imshow("YOLOv8 Fire Detection - Image Preview", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_stream(
    detector: FireSmokeDetector,
    source: str,
    alarm: HardwareAlarmController,
    save_path: str = None,
    view: bool = True
):
    import cv2

    # Identify if source is webcam index or file path
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise VideoStreamError(f"Failed to open video source: {source}")

    logger.info(f"Started video stream processing on source: {source}")

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("Video stream completed or interrupted.")
                break

            # Calculate FPS
            curr_time = time.time()
            fps_val = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
            prev_time = curr_time

            # Inference
            detections = detector.predict_frame(frame)
            has_fire = any("fire" in d["label"].lower() for d in detections)
            
            # Trigger Hardware Alarm if connected
            alarm.trigger_alarm(has_fire)

            # Annotate
            annotated = detector.annotate_frame(frame, detections, show_fps=fps_val)

            if writer:
                writer.write(annotated)

            if view:
                cv2.imshow("YOLOv8 Fire & Smoke Detection System", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested exit ('q' pressed).")
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        if view:
            cv2.destroyAllWindows()
        alarm.close()


def main():
    args = parse_args()
    logger.info("Initializing YOLOv8 Fire & Smoke Detection Pipeline...")

    try:
        detector = FireSmokeDetector(
            weights_path=args.weights,
            conf_threshold=args.conf,
            device=args.device
        )
        alarm = HardwareAlarmController(port=args.arduino_port)

        # Check source type
        if args.source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            process_image(detector, args.source, save_path=args.save, view=args.view)
        else:
            process_stream(detector, args.source, alarm, save_path=args.save, view=args.view)

    except Exception as e:
        logger.error(f"Execution terminated with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
