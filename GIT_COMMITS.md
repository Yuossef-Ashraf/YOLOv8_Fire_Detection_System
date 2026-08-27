# Suggested Conventional Git Commits for YOLOv8_Fire_Detection_System

```bash
# 1. Project Dependencies & Configuration
git add requirements.txt .gitignore
git commit -m "chore(deps): configure pinned dependencies and ignore patterns"

# 2. Modular Engine & Hardware Interface
git add detector.py detect.py logging_config.py exceptions.py
git commit -m "feat(core): implement modular FireSmokeDetector and CLI inference runner with Arduino alarm"

# 3. Test Suite
git add tests/
git commit -m "test(detector): add pytest test suite covering model inference and serial alarm"

# 4. CI/CD Pipeline
git add .github/
git commit -m "ci(actions): add automated inference checks and flake8 workflow"

# 5. Technical Documentation
git add docs/ CHANGELOG.md CONTRIBUTING.md
git commit -m "docs(arch): add system architecture and contribution guidelines"

# 6. Master README Update
git add README.md
git commit -m "docs(readme): overhaul README with real-time metrics, system architecture, and quickstart"
```
