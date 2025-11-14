# ConfMap GUI Processor

A Windows GUI application for converting configuration files between different formats using the confmap library.

## Features

- Convert between YAML, JSON, INI, and other configuration formats
- Modern GUI with before/after comparison
- Support for long filenames with non-ANSI characters
- Threaded processing for responsive UI
- Single executable file for easy distribution

## Download

Go to the **Actions** tab in this repository, click on the latest successful workflow run, and download the `ConfMapProcessor-Windows.zip` artifact.

## Building from Source

The application is automatically built using GitHub Actions. To build manually:

1. Install Python 3.9+
2. Install dependencies: `pip install -r requirements.txt`
3. Build with PyInstaller: `pyinstaller --name=ConfMapProcessor --windowed --onefile confmap_gui.py`

## Usage

1. Download the executable from the latest release
2. Run `ConfMapProcessor.exe`
3. Load a configuration file using the "Load Configuration File" button
4. Click "Process Configuration" to convert the file
5. Save the result using "Save Processed Result"

## Supported Formats

- YAML (.yaml, .yml)
- JSON (.json)
- INI (.ini, .cfg, .conf)
- TOML (.toml)
- And more via the confmap library
