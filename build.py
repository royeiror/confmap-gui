# build.py
import sys
import os
from PyInstaller.__main__ import run

if __name__ == '__main__':
    opts = [
        'confmap_gui.py',
        '--name=ConfMapProcessor',
        '--windowed',  # No console window
        '--onefile',   # Single executable
        '--icon=icon.ico',  # Optional: add an icon
        '--add-data=templates;templates',  # If confmap needs templates
        '--clean',
        '--noconfirm'
    ]
    
    run(opts)
