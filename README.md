# ConfMap 3D Processor

A Windows GUI application for 3D mesh processing and conformal mapping using the confmap library.

## Features

- Process 3D OBJ files with conformal mapping
- Multiple algorithms: BFF, SCP, AE
- Generate UV coordinates for texture mapping
- Real-time progress logging
- Support for large 3D meshes

## Supported Methods

- **BFF**: Boundary First Flattening - Fast and robust for meshes with boundaries
- **SCP**: Spectral Conformal Parameterization - Good for closed meshes  
- **AE**: Authalic Embedding - Preserves area relationships

## Usage

1. Download the executable from the latest release
2. Run `ConfMap3DProcessor.exe`
3. Load an OBJ file using "Load OBJ File"
4. Set output location (optional)
5. Select processing method and options
6. Click "Process Mesh" to generate conformal map
7. The output OBJ will contain UV coordinates for texturing

## Input/Output

- **Input**: Standard .obj files (vertices and faces)
- **Output**: .obj files with added UV coordinates
- Preserves original geometry while adding UV mapping

## Building

The application is automatically built using GitHub Actions. To build manually:

1. Install Python 3.9+
2. Install dependencies: `pip install -r requirements.txt`
3. Build with PyInstaller: `pyinstaller --name=ConfMap3DProcessor --windowed --onefile confmap_gui.py`
