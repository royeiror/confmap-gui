@echo off
echo Building ConfMap GUI Application...

pip install -r requirements.txt

echo Running PyInstaller...
pyinstaller --name=ConfMapProcessor ^
            --windowed ^
            --onefile ^
            --clean ^
            --noconfirm ^
            confmap_gui.py

echo Build complete! Check the 'dist' folder.
pause
