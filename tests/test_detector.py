"""
Unit tests for YOLOv8 detector and hardware alarm controller.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

import detector
from exceptions import ModelNotFoundError, ModelInferenceError, FireDetectionError


class TestHardwareAlarmController:
    """Test serial communication and safety fallbacks."""

    def test_alarm_without_port_is_noop(self):
        alarm = detector.HardwareAlarmController(port=None)
        assert alarm._is_active is False
        # Calling trigger must not raise errors
        alarm.trigger_alarm(True)
        alarm.trigger_alarm(False)
        alarm.close()

    @patch("serial.Serial")
    def test_alarm_trigger_with_serial(self, mock_serial):
        mock_instance = MagicMock()
        mock_instance.is_open = True
        mock_serial.return_value = mock_instance

        alarm = detector.HardwareAlarmController(port="COM6", baudrate=9600)
        alarm.trigger_alarm(True)
        mock_instance.write.assert_called_with(b'1')

        alarm.trigger_alarm(False)
        mock_instance.write.assert_called_with(b'0')
        alarm.close()


class TestFireSmokeDetector:
    """Test model loading, inference routing, and annotation."""

    def test_model_not_found_raises_exception(self):
        with pytest.raises(ModelNotFoundError):
            detector.FireSmokeDetector(weights_path="non_existent_fire_weights_99.pt")

    @patch("detector.FireSmokeDetector._load_model")
    def test_predict_frame_parsing(self, mock_load, mock_yolo_detection_result):
        det = detector.FireSmokeDetector(weights_path="fire.pt")
        mock_model = MagicMock()
        mock_model.return_value = mock_yolo_detection_result
        det.model = mock_model
        det.class_names = {0: "Fire", 1: "Smoke"}

        # Simulate frame
        frame = [[0] * 100] * 100
        results = det.predict_frame(frame)

        assert len(results) == 2
        assert results[0]["label"] == "Fire"
        assert results[0]["confidence"] == 0.95
        assert results[1]["label"] == "Smoke"
        assert results[1]["confidence"] == 0.88
