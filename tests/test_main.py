"""
Main integration and CLI tests for YOLOv8 Fire Detection System.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

import detect
import logging_config


class TestSystemIntegration:
    """System level integration tests."""

    def test_logging_setup(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        logger = logging_config.setup_logging(log_dir=log_dir, log_file_name="test_fire.log")
        assert logger is not None
        assert os.path.exists(log_dir)

    @patch("detector.FireSmokeDetector._load_model")
    @patch("cv2.imread")
    @patch("cv2.imshow")
    @patch("cv2.waitKey")
    @patch("cv2.destroyAllWindows")
    def test_process_image_flow(self, mock_destroy, mock_wait, mock_imshow, mock_imread, mock_load, mock_yolo_detection_result):
        import detector
        det = detector.FireSmokeDetector(weights_path="fire.pt")
        mock_model = MagicMock()
        mock_model.return_value = mock_yolo_detection_result
        det.model = mock_model
        det.class_names = {0: "Fire", 1: "Smoke"}

        mock_imread.return_value = MagicMock()
        
        # Test image processing flow
        detect.process_image(det, "sample_test.png", view=False)
        assert det.model.called
