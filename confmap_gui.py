import sys
import os
import json
import tempfile
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QLabel, QFileDialog, 
                             QWidget, QSplitter, QMessageBox, QProgressBar,
                             QGroupBox, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor
import confmap

class ConfMapWorker(QThread):
    """Worker thread for confmap processing to prevent GUI freezing"""
    finished = pyqtSignal(str, str)  # original, processed
    error = pyqtSignal(str)
    
    def __init__(self, config_text, use_json=False):
        super().__init__()
        self.config_text = config_text
        self.use_json = use_json
    
    def run(self):
        try:
            # Create temporary files for processing
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(self.config_text)
                temp_input = f.name
            
            # Process with confmap
            if self.use_json:
                result = confmap.from_file(temp_input, to='json')
            else:
                result = confmap.from_file(temp_input, to='yaml')
            
            # Clean up temp file
            os.unlink(temp_input)
            
            self.finished.emit(self.config_text, result)
            
        except Exception as e:
            self.error.emit(str(e))

class ConfMapGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("ConfMap Processor - Configuration File Converter")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("ConfMap Configuration File Processor")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Convert between different configuration file formats (YAML, JSON, INI, etc.)")
        desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc)
        
        # Options group
        options_group = QGroupBox("Conversion Options")
        options_layout = QHBoxLayout(options_group)
        
        self.json_checkbox = QCheckBox("Convert to JSON")
        self.json_checkbox.setChecked(False)
        options_layout.addWidget(self.json_checkbox)
        
        options_layout.addStretch()
        layout.addWidget(options_group)
        
        # File selection buttons
        file_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Configuration File")
        self.load_btn.clicked.connect(self.load_file)
        file_layout.addWidget(self.load_btn)
        
        self.save_btn = QPushButton("Save Processed Result")
        self.save_btn.clicked.connect(self.save_file)
        self.save_btn.setEnabled(False)
        file_layout.addWidget(self.save_btn)
        
        self.process_btn = QPushButton("Process Configuration")
        self.process_btn.clicked.connect(self.process_config)
        self.process_btn.setEnabled(False)
        file_layout.addWidget(self.process_btn)
        
        layout.addLayout(file_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Splitter for before/after views
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - original
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(QLabel("Original Configuration:"))
        self.original_text = QTextEdit()
        self.original_text.setPlaceholderText("Original configuration will appear here...")
        left_layout.addWidget(self.original_text)
        splitter.addWidget(left_widget)
        
        # Right panel - processed
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(QLabel("Processed Configuration:"))
        self.processed_text = QTextEdit()
        self.processed_text.setPlaceholderText("Processed configuration will appear here...")
        self.processed_text.setReadOnly(True)
        right_layout.addWidget(self.processed_text)
        splitter.addWidget(right_widget)
        
        splitter.setSizes([600, 600])
        layout.addWidget(splitter, 1)
        
        # Status bar
        self.statusBar().showMessage("Ready to load configuration file")
        
        # Initialize worker
        self.worker = None
        self.current_processed_content = ""
        
    def load_file(self):
        """Load a configuration file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Configuration File",
            "",
            "Configuration Files (*.yaml *.yml *.json *.ini *.conf *.cfg *.toml);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.original_text.setText(content)
                self.process_btn.setEnabled(True)
                self.save_btn.setEnabled(False)
                self.statusBar().showMessage(f"Loaded: {file_path}")
                
            except UnicodeDecodeError:
                # Try with different encoding
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                    self.original_text.setText(content)
                    self.process_btn.setEnabled(True)
                    self.save_btn.setEnabled(False)
                    self.statusBar().showMessage(f"Loaded with alternative encoding: {file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Could not read file: {str(e)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not read file: {str(e)}")
    
    def process_config(self):
        """Process the configuration using confmap"""
        content = self.original_text.toPlainText().strip()
        if not content:
            QMessageBox.warning(self, "Warning", "No content to process!")
            return
        
        # Disable buttons during processing
        self.process_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Start worker thread
        self.worker = ConfMapWorker(content, self.json_checkbox.isChecked())
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.start()
        
        self.statusBar().showMessage("Processing configuration...")
    
    def on_processing_finished(self, original, processed):
        """Handle successful processing"""
        self.processed_text.setText(processed)
        self.current_processed_content = processed
        
        # Re-enable buttons
        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.statusBar().showMessage("Processing completed successfully!")
        
        # Show success message
        QMessageBox.information(self, "Success", "Configuration processed successfully!")
    
    def on_processing_error(self, error_msg):
        """Handle processing errors"""
        # Re-enable buttons
        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.statusBar().showMessage(f"Error: {error_msg}")
        
        # Show error message
        QMessageBox.critical(self, "Processing Error", 
                           f"An error occurred during processing:\n{error_msg}")
    
    def save_file(self):
        """Save the processed configuration"""
        if not self.current_processed_content:
            QMessageBox.warning(self, "Warning", "No processed content to save!")
            return
        
        # Determine default file extension
        ext = ".json" if self.json_checkbox.isChecked() else ".yaml"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processed Configuration",
            "",
            f"Configuration Files (*{ext});;All Files (*)",
            options=QFileDialog.Options()
        )
        
        if file_path:
            try:
                # Ensure proper encoding for non-ANSI characters
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.current_processed_content)
                
                self.statusBar().showMessage(f"Saved: {file_path}")
                QMessageBox.information(self, "Success", f"File saved successfully!\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save file: {str(e)}")

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("ConfMap Processor")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("ConfMap")
    
    window = ConfMapGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
