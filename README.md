# Robot Agent Setup Guide

This project captures a drawing/photo, extracts a person-like skeleton from the image using a vision model API, and animates a dancing figure with Pygame.

## 1) Requirements

- Python 3.10+ (recommended: 3.11)
- Webcam (for live capture)
- Windows/macOS/Linux with GUI support (required for OpenCV and Pygame windows)

## 2) Clone and open project

```bash
git clone https://github.com/Bier0003/robot_agent.git
cd robot_agent
```

## 3) Create and activate a virtual environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4) Install dependencies

This project currently does not include a `requirements.txt`, so install directly:

```bash
pip install opencv-python numpy pygame requests python-dotenv openai
```

## 5) Configure environment variables

Create or edit `.env` in the project root:

```env
Agent_API_Key=your_api_key_here
Agent_URL=https://openrouter.ai/api/v1/chat/completions
Agent_MODEL=google/gemini-3.1-flash
```

Notes:
- `Agent_API_Key` and `Agent_URL` are required.
- `Agent_MODEL` is optional in code, but recommended to set explicitly.
- Do not commit real API keys to git.

## 6) Run the app

```bash
python main.py
```

Flow:
1. Camera window opens.
2. Press `Space` to capture image, or `Esc` to cancel.
3. Background is cut from the captured figure.
4. Vision model extracts joints.
5. Pygame opens and runs dance animation.

## 7) Quick environment check (optional)

```bash
python test_pygame.py
```

This prints installed library versions to verify setup.

## Troubleshooting

- Camera cannot open:
  - Close other apps using webcam.
  - Check OS camera permissions.
- API request fails (400/401):
  - Verify `Agent_API_Key`, `Agent_URL`, and `Agent_MODEL` in `.env`.
- Pygame/OpenCV window does not show:
  - Make sure you are running in a desktop GUI session (not headless).
  - Update GPU/video drivers if needed.
- `ModuleNotFoundError`:
  - Confirm virtual environment is activated and dependencies are installed.

## Project files

- `main.py`: Main app flow.
- `capture_picture.py`: Captures image from webcam.
- `rigger_figure.py`: Removes background and creates transparent figure PNG.
- `vision_agent_module.py`: Calls vision API to extract joints.
- `dance_module.py`: Animates skeleton/figure with Pygame.
- `test_pygame.py`: Basic dependency/version check.
