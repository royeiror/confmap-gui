# build.py - Optional script for manual building
import PyInstaller.__main__

PyInstaller.__main__.run([
    'confmap_gui.py',
    '--name=ConfMapProcessor',
    '--windowed',
    '--onefile',
    '--clean',
    '--noconfirm'
])
