# Contributing to YOLOv8 Fire & Smoke Detection System

Thank you for your interest in contributing to the **YOLOv8 Fire & Smoke Detection System**! We welcome contributions to enhance model performance, inference speed, dataset curation, alerting integrations, and documentation.

---

## How to Contribute

Please follow these numbered steps to submit your contributions:

1. **Fork the Repository**: Click the **Fork** button at the top right of the repository page on GitHub.
2. **Clone Your Fork**: Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOLOv8_Fire_Detection_System.git
   cd YOLOv8_Fire_Detection_System
   ```
3. **Create a Feature Branch**: Create a descriptive branch for your work:
   ```bash
   git checkout -b feature/your-feature
   ```
4. **Make Your Changes**: Implement your model improvements, pipeline updates, or documentation fixes.
5. **Write Tests**: Add tests covering inference and validation logic in the `tests/` directory.
6. **Commit Your Changes**: Follow standard commit conventions:
   ```bash
   git commit -m "feat: add real-time alert trigger on detection"
   ```
7. **Push to Your Fork**: Push your branch to GitHub:
   ```bash
   git push origin feature/your-feature
   ```
8. **Open a Pull Request**: Submit a Pull Request (PR) against the `main` branch with detailed descriptions of changes and validation results.

---

## Branch Naming Convention

- `feature/your-feature` — New features (e.g., `feature/rtsp-stream-input`, `feature/sound-alarm`)
- `fix/bug-name` — Bug fixes (e.g., `fix/bounding-box-scaling`)
- `docs/what-you-documented` — Documentation updates (e.g., `docs/gpu-setup-guide`)
- `refactor/scope` — Refactoring inference scripts or pipelines
- `test/test-scope` — Adding or updating test cases

---

## Commit Message Format

Follow Conventional Commits:

```
<type>: <short summary in present tense>

[optional body with details]
```

### Examples:
- `feat: add video stream inference script`
- `fix: handle corrupted input frames gracefully`
- `docs: update README with new mAP benchmarks`
- `test: add detector inference unit tests`

---

## Code Style Guidelines

- **PEP 8 Compliance**: Follow PEP 8 guidelines.
- **Line Length**: Maximum line length of **88 characters**.
- **Formatter**: Format code with `black`:
  ```bash
  pip install black
  black .
  ```
- **Type Annotations**: Use Python type hints wherever possible.

---

## Testing Requirements

- Ensure all scripts and tests execute cleanly:
  ```bash
  pytest tests/ -v
  ```
- Verify model inference runs correctly with `fire.pt` weights before submitting PRs.

---

## Reporting Bugs

If you find a bug or performance degradation, open an Issue on GitHub with:
1. **Python Version & PyTorch / Ultralytics Versions**: (e.g., Python 3.10, PyTorch 2.3.0, Ultralytics 8.2.18)
2. **Hardware Info**: CPU model and GPU model (with CUDA version if applicable)
3. **Steps to Reproduce**: Clear step-by-step instructions
4. **Expected vs Actual Output**
5. **Full Error Traceback**: Complete log inside triple backticks (```)
