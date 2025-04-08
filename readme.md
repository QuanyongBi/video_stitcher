# Video Stitcher

A Python application for creating ultra-high-resolution images from video captures.

## Project Overview

This tool processes multiple video captures of a scene and stitches them together to create a single ultra-high-resolution image. The application implements several computer vision techniques including:

- Video frame extraction
- Feature detection and matching
- Image alignment and transformation
- Seamless image blending
- Interactive image viewer with zoom capabilities

## Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)
- pip (Python package installer)

## Project Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/video-stitcher.git
cd video-stitcher
```

### 2. Create a Virtual Environment

We use Python's built-in `venv` module to create an isolated environment:

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

You'll see `(venv)` appear at the beginning of your command prompt, indicating the environment is active.

### 3. Install Dependencies

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
pip install -e .
```

The `-e` flag installs the package in development mode, allowing you to modify the code without reinstalling.

### 4. VS Code Configuration

If you're using VS Code (recommended):

1. Open the project folder in VS Code
2. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
3. Type "Python: Select Interpreter" and select that option
4. Choose the interpreter from your virtual environment (it should show something like `'venv':venv`)

VS Code will now use the correct Python interpreter with all installed dependencies.

## Project Structure

```
video-stitcher/
├── requirements.txt     # Project dependencies
├── setup.py            # Package configuration
├── README.md           # This file
├── src/                # Source code
│   └── video_stitcher/ # Main package
│       └── __init__.py # Package initialization
└── tests/              # Unit tests
    └── __init__.py     # Test package initialization
```

## Development Workflow

1. Ensure your virtual environment is activated before working on the project
2. Run tests with `pytest` from the project root
3. Install any new dependencies with `pip install <package>` and then add them to `requirements.txt`

## Deactivating the Environment

When you're done working on the project, you can deactivate the virtual environment:

```bash
deactivate
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
