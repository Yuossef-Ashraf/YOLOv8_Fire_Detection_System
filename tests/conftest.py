"""
Pytest configuration and fixtures for YOLOv8 Fire & Smoke Detection System.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def mock_yolo_detection_result():
    """Mock YOLO prediction output structure."""
    class MockBox:
        def __init__(self, xyxy, conf, cls):
            self.xyxy = [xyxy]
            self.conf = [conf]
            self.cls = [cls]

    class MockResult:
        def __init__(self):
            self.boxes = [
                MockBox([50, 60, 200, 220], 0.95, 0),  # Fire
                MockBox([220, 40, 350, 180], 0.88, 1)  # Smoke
            ]

    return [MockResult()]


@pytest.fixture
def sample_test_image(tmp_path):
    """Generate a clean synthetic test image for CLI tests."""
    img_path = str(tmp_path / "test_fire_scene.png")
    # Minimal 1x1 valid PNG binary fallback
    minimal_png = (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06'
        b'\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02'
        b'\xfe\xa7Cv\x9a\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    with open(img_path, "wb") as f:
        f.write(minimal_png)
    return img_path
