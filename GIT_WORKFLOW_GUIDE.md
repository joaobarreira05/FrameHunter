# FrameHunter: Git Workflow & Pipeline Guide

This guide explains how to contribute to FrameHunter and how the CI/CD pipeline works.

## 1. Project Structure & Branches
- **`main`**: The stable branch. Only production-ready code should be here.
- **`feature/xxx`**: Work in progress for new features.
- **`fix/xxx`**: Bug fixes.

## 2. Contribution Flow (PRs)
To add a new feature or fix a bug:
1. **Fork** the repository on GitHub.
2. **Clone** your fork locally.
3. **Branch**: `git checkout -b feature/my-cool-feature`.
4. **Code**: Implement your changes.
5. **Test**: Run `python -m pytest` to ensure no regressions.
6. **Commit**: Use clear messages (e.g., `feat: add exhaustive frame scan`).
7. **Push**: `git push origin feature/my-cool-feature`.
8. **PR**: Open a Pull Request from your fork's branch to FrameHunter's `main`.

## 3. Automation Pipeline (GitHub Actions)
The project uses GitHub Actions for automation:
- **`test-lint.yml`**: Runs on every Push or PR to `main`. It checks code style (linting) and runs the test suite.
- **`release.yml`**: Triggers when a new **Tag** starting with `v*` (e.g., `v0.2.0`) is pushed. It:
    - Creates a GitHub Release.
    - Builds a Docker image.
    - Uploads the release artifacts.

## 4. Releasing a New Version
1. Ensure `main` is up to date and tests pass.
2. Create a tag: `git tag v0.2.0`.
3. Push the tag: `git push origin v0.2.0`.
4. The pipeline will automatically create the release page and build the Docker image.

## 5. Local Setup for Developers
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m pytest
```

---
*Note: This document is for internal reference and is ignored by Git.*
