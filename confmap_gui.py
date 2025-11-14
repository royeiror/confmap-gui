import sys
import os
import tempfile
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QLabel, QFileDialog, 
                             QWidget, QSplitter, QMessageBox, QProgressBar,
                             QGroupBox, QComboBox, QCheckBox, QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor

# Import our minimal confmap implementation
from minimal_confmap import BFF, SCP, AE, read_obj, write_obj
from mesh_viewer import MeshViewer

class ConfMapWorker(QThread):
    """Worker thread for confmap processing to prevent GUI freezing"""
    finished = pyqtSignal(str, str, str, object, object, object, object)  # input_path, output_path, log, vertices, faces, uv_vertices, uv_faces
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, input_path, output_path, method="BFF", generate_uv=True):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.method = method
        self.generate_uv = generate_uv
        self.vertices = None
        self.faces = None
        self.uv_vertices = None
        self.uv_faces = None
    
    def run(self):
        try:
            self.progress.emit("Reading OBJ file...")
            
            # Read the input OBJ file
            self.vertices, self.faces = read_obj(self.input_path)
            
            self.progress.emit(f"Loaded mesh: {len(self.vertices)} vertices, {len(self.faces)} faces")
            
            # Select the conformal mapping method
            if self.method == "BFF":
                self.progress.emit("Using BFF (Boundary First Flattening) method...")
                cm = BFF(self.vertices, self.faces)
            elif self.method == "SCP":
                self.progress.emit("Using SCP (Spectral Conformal Parameterization) method...")
                cm = SCP(self.vertices, self.faces)
            elif self.method == "AE":
                self.progress.emit("Using AE (Authalic Embedding) method...")
                cm = AE(self.vertices, self.faces)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            self.progress.emit("Computing conformal map...")
            
            # Generate the UV layout
            if self.generate_uv:
                result = cm.layout()
                self.uv_vertices = result.uv_vertices
                self.uv_faces = result.uv_faces
                self.progress.emit("UV layout generated successfully")
                
                # Write output with UV coordinates
                write_obj(self.output_path, self.vertices, self.faces, self.uv_vertices, self.uv_faces)
                log_message = f"""Processing Complete!
Input: {self.input_path}
Output: {self.output_path}
Method: {self.method}
Vertices: {len(self.vertices)}
Faces: {len(self.faces)}
UV vertices: {len(self.uv_vertices)}

The output file contains the original 3D mesh with UV coordinates for texture mapping."""
            else:
                # Just write the original mesh (for debugging)
                write_obj(self.output_path, self.vertices, self.faces)
                log_message = f"""Processing Complete (No UV generated)
Input: {self.input_path}
Output: {self.output_path}
Method: {self.method}
Vertices: {len(self.vertices)}
Faces: {len(self.faces)}"""
            
            self.finished.emit(self.input_path, self.output_path, log_message, 
                             self.vertices, self.faces, self.uv_vertices, self.uv_faces)
            
        except Exception as e:
            self.error.emit(str(e))

class ConfMapGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_file = None
        self.output_file = None
        self.current_vertices = None
        self.current_faces = None
        self.current_uv_vertices = None
        self.current_uv_faces = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("ConfMap Processor - 3D Mesh Conformal Mapping")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("3D Mesh Conformal Map Processor")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Process 3D OBJ files to generate conformal maps for UV unwrapping")
        desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc)
        
        # Options group
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout(options_group)
        
        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Conformal Mapping Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["BFF", "SCP", "AE"])
        self.method_combo.setToolTip("BFF: Boundary First Flattening (Recommended)\nSCP: Spectral Conformal Parameterization\nAE: Authalic Embedding")
        method_layout.addWidget(self.method_combo)
        method_layout.addStretch()
        options_layout.addLayout(method_layout)
        
        # UV generation option
        self.uv_checkbox = QCheckBox("Generate UV coordinates")
        self.uv_checkbox.setChecked(True)
        self.uv_checkbox.setToolTip("Generate UV coordinates for texture mapping")
        options_layout.addWidget(self.uv_checkbox)
        
        layout.addWidget(options_group)
        
        # File selection buttons
        file_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load OBJ File")
        self.load_btn.clicked.connect(self.load_file)
        self.load_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        file_layout.addWidget(self.load_btn)
        
        self.output_btn = QPushButton("Set Output Location")
        self.output_btn.clicked.connect(self.set_output)
        self.output_btn.setEnabled(False)
        file_layout.addWidget(self.output_btn)
        
        self.process_btn = QPushButton("Process Mesh")
        self.process_btn.clicked.connect(self.process_mesh)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("QPushButton { font-weight: bold; background-color: #4CAF50; color: white; }")
        file_layout.addWidget(self.process_btn)
        
        layout.addLayout(file_layout)
        
        # File info
        info_layout = QHBoxLayout()
        
        self.input_label = QLabel("No file selected")
        self.input_label.setWordWrap(True)
        info_layout.addWidget(QLabel("Input:"))
        info_layout.addWidget(self.input_label, 1)
        
        self.output_label = QLabel("No output location set")
        self.output_label.setWordWrap(True)
        info_layout.addWidget(QLabel("Output:"))
        info_layout.addWidget(self.output_label, 1)
        
        layout.addLayout(info_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Progress log
        self.progress_label = QLabel("Ready")
        self.progress_label.setWordWrap(True)
        layout.addWidget(self.progress_label)
        
        # Create tab widget for visualization
        self.tabs = QTabWidget()
        
        # Log tab
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        self.log_text = QTextEdit()
        self.log_text.setPlaceholderText("Processing log will appear here...")
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        self.tabs.addTab(log_widget, "Processing Log")
        
        # Visualization tab
        vis_widget = QWidget()
        vis_layout = QVBoxLayout(vis_widget)
        
        # Visualization controls
        vis_controls_layout = QHBoxLayout()
        
        vis_controls_layout.addWidget(QLabel("View Mode:"))
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["3D Mesh", "UV Layout"])
        self.view_mode_combo.currentTextChanged.connect(self.on_view_mode_changed)
        vis_controls_layout.addWidget(self.view_mode_combo)
        
        self.wireframe_checkbox = QCheckBox("Wireframe")
        self.wireframe_checkbox.stateChanged.connect(self.on_wireframe_changed)
        vis_controls_layout.addWidget(self.wireframe_checkbox)
        
        vis_controls_layout.addStretch()
        vis_layout.addLayout(vis_controls_layout)
        
        # Create mesh viewer
        self.mesh_viewer = MeshViewer()
        vis_layout.addWidget(self.mesh_viewer, 1)
        
        self.tabs.addTab(vis_widget, "3D Visualization")
        
        layout.addWidget(self.tabs, 1)
        
        # Status bar
        self.statusBar().showMessage("Ready to load OBJ file")
        
        # Initialize worker
        self.worker = None
        
    def on_view_mode_changed(self, mode):
        """Handle view mode change"""
        self.mesh_viewer.set_display_mode(mode == "3D Mesh")
        
    def on_wireframe_changed(self, state):
        """Handle wireframe toggle"""
        self.mesh_viewer.set_wireframe(state == Qt.Checked)
        
    def load_file(self):
        """Load an OBJ file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select OBJ File",
            "",
            "3D Model Files (*.obj);;All Files (*)"
        )
        
        if file_path:
            try:
                self.input_file = file_path
                self.input_label.setText(Path(file_path).name)
                self.input_label.setToolTip(file_path)
                
                # Read and display the mesh immediately
                vertices, faces = read_obj(file_path)
                self.current_vertices = vertices
                self.current_faces = faces
                self.mesh_viewer.set_mesh(vertices, faces)
                
                # Set default output path
                input_path = Path(file_path)
                default_output = input_path.parent / f"{input_path.stem}_with_uv{input_path.suffix}"
                self.output_file = str(default_output)
                self.output_label.setText(default_output.name)
                self.output_label.setToolTip(str(default_output))
                
                self.output_btn.setEnabled(True)
                self.process_btn.setEnabled(True)
                self.statusBar().showMessage(f"Loaded: {file_path}")
                
                # Clear previous log
                self.log_text.clear()
                self.log_text.append(f"Loaded OBJ file: {file_path}")
                self.log_text.append(f"Mesh statistics: {len(vertices)} vertices, {len(faces)} faces")
                
                # Switch to visualization tab
                self.tabs.setCurrentIndex(1)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load file: {str(e)}")
            
    def set_output(self):
        """Set output file location"""
        if not self.input_file:
            return
            
        default_name = Path(self.input_file).stem + "_with_uv.obj"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processed OBJ File",
            str(Path(self.input_file).parent / default_name),
            "OBJ Files (*.obj);;All Files (*)"
        )
        
        if file_path:
            self.output_file = file_path
            self.output_label.setText(Path(file_path).name)
            self.output_label.setToolTip(file_path)
            self.statusBar().showMessage(f"Output set to: {file_path}")
            
    def process_mesh(self):
        """Process the 3D mesh using confmap"""
        if not self.input_file or not self.output_file:
            QMessageBox.warning(self, "Warning", "Please select both input and output files!")
            return
        
        if not os.path.exists(self.input_file):
            QMessageBox.critical(self, "Error", "Input file does not exist!")
            return
        
        # Disable buttons during processing
        self.process_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.output_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Get processing options
        method = self.method_combo.currentText()
        generate_uv = self.uv_checkbox.isChecked()
        
        # Start worker thread
        self.worker = ConfMapWorker(self.input_file, self.output_file, method, generate_uv)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.progress.connect(self.on_progress_update)
        self.worker.start()
        
        self.statusBar().showMessage("Processing 3D mesh...")
        self.log_text.append(f"Starting processing with {method} method...")
        
    def on_progress_update(self, message):
        """Update progress messages"""
        self.progress_label.setText(message)
        self.log_text.append(f"• {message}")
        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
        
    def on_processing_finished(self, input_path, output_path, log_message, vertices, faces, uv_vertices, uv_faces):
        """Handle successful processing"""
        # Store the processed data
        self.current_vertices = vertices
        self.current_faces = faces
        self.current_uv_vertices = uv_vertices
        self.current_uv_faces = uv_faces
        
        # Update the mesh viewer with both 3D and UV data
        self.mesh_viewer.set_mesh(vertices, faces, uv_vertices, uv_faces)
        
        # Re-enable buttons
        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.output_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.statusBar().showMessage("Processing completed successfully!")
        self.progress_label.setText("Processing completed!")
        
        # Add final log message
        self.log_text.append("\n" + "="*50)
        self.log_text.append("PROCESSING COMPLETED SUCCESSFULLY!")
        self.log_text.append("="*50)
        self.log_text.append(log_message)
        
        # Switch to visualization tab to see the result
        self.tabs.setCurrentIndex(1)
        
        # Show success message
        QMessageBox.information(self, "Success", 
                              f"3D mesh processed successfully!\n\n"
                              f"Input: {Path(input_path).name}\n"
                              f"Output: {Path(output_path).name}\n\n"
                              f"Switch to the '3D Visualization' tab to see the results!")
        
    def on_processing_error(self, error_msg):
        """Handle processing errors"""
        # Re-enable buttons
        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.output_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.statusBar().showMessage(f"Error: {error_msg}")
        self.progress_label.setText("Processing failed!")
        
        # Add error to log
        self.log_text.append("\n" + "="*50)
        self.log_text.append("PROCESSING FAILED!")
        self.log_text.append("="*50)
        self.log_text.append(f"Error: {error_msg}")
        
        # Show error message
        QMessageBox.critical(self, "Processing Error", 
                           f"An error occurred during processing:\n{error_msg}")

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("ConfMap 3D Processor")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("ConfMap")
    
    window = ConfMapGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
