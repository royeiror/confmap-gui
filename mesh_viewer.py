import numpy as np
import xml.etree.ElementTree as ET
from PyQt5.QtWidgets import (QOpenGLWidget, QVBoxLayout, QHBoxLayout, QWidget, 
                             QLabel, QSplitter, QCheckBox, QPushButton, QComboBox,
                             QFileDialog, QMessageBox, QSlider, QSpinBox, QApplication,
                             QProgressBar, QTextEdit, QTabWidget, QGroupBox)
from PyQt5.QtCore import Qt, QPoint, QThread, pyqtSignal
from PyQt5.QtGui import QMouseEvent
from OpenGL.GL import *
from OpenGL.GLU import *
import os
import tempfile
import sys
import trimesh
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import time

class ConformalMappingThread(QThread):
    """Thread for computing conformal mapping to avoid GUI freezing"""
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object, object)  # uv_vertices, uv_faces
    
    def __init__(self, vertices, faces):
        super().__init__()
        self.vertices = vertices
        self.faces = faces
        
    def run(self):
        try:
            self.log_signal.emit("Starting conformal mapping...")
            uv_vertices, uv_faces = self.compute_conformal_map()
            self.finished_signal.emit(uv_vertices, uv_faces)
            self.log_signal.emit("Conformal mapping completed successfully!")
        except Exception as e:
            self.log_signal.emit(f"Error in conformal mapping: {str(e)}")
            self.finished_signal.emit(None, None)
    
    def compute_conformal_map(self):
        """Compute a simple conformal map using harmonic mapping"""
        self.progress_signal.emit(10)
        self.log_signal.emit("Building mesh connectivity...")
        
        mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        
        # Find boundary vertices
        boundaries = mesh.boundary_loops
        if len(boundaries) == 0:
            self.log_signal.emit("No boundary found - using convex hull as boundary")
            # Use convex hull as boundary for closed meshes
            hull = mesh.convex_hull
            boundaries = hull.boundary_loops
        
        if len(boundaries) == 0:
            raise ValueError("Cannot find suitable boundary for mapping")
        
        boundary = boundaries[0]  # Use first boundary
        self.progress_signal.emit(30)
        self.log_signal.emit(f"Found boundary with {len(boundary)} vertices")
        
        # Map boundary to circle
        n_boundary = len(boundary)
        boundary_uv = np.zeros((n_boundary, 2))
        for i, vertex_idx in enumerate(boundary):
            angle = 2 * np.pi * i / n_boundary
            boundary_uv[i] = [0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)]
        
        self.progress_signal.emit(50)
        self.log_signal.emit("Setting up linear system...")
        
        # Build cotangent Laplacian matrix
        n_vertices = len(self.vertices)
        L = lil_matrix((n_vertices, n_vertices))
        
        for face in self.faces:
            if len(face) == 3:
                for i in range(3):
                    v1 = face[i]
                    v2 = face[(i + 1) % 3]
                    v3 = face[(i + 2) % 3]
                    
                    # Compute cotangent weight
                    vec1 = self.vertices[v1] - self.vertices[v3]
                    vec2 = self.vertices[v2] - self.vertices[v3]
                    cot_angle = np.dot(vec1, vec2) / np.linalg.norm(np.cross(vec1, vec2))
                    
                    L[v1, v2] += cot_angle
                    L[v2, v1] += cot_angle
                    L[v1, v1] -= cot_angle
                    L[v2, v2] -= cot_angle
        
        self.progress_signal.emit(70)
        self.log_signal.emit("Solving linear system...")
        
        # Solve for interior vertices
        uv_vertices = np.zeros((n_vertices, 2))
        
        # Set boundary conditions
        for coord in [0, 1]:  # x and y coordinates
            b = np.zeros(n_vertices)
            
            # Set boundary values
            for i, vertex_idx in enumerate(boundary):
                b[vertex_idx] = boundary_uv[i, coord]
            
            # Modify system for boundary conditions
            L_mod = L.copy().tocsr()
            b_mod = b.copy()
            
            for vertex_idx in boundary:
                L_mod[vertex_idx, :] = 0
                L_mod[vertex_idx, vertex_idx] = 1
                b_mod[vertex_idx] = b[vertex_idx]
            
            # Solve
            uv_vertices[:, coord] = spsolve(L_mod, b_mod)
        
        self.progress_signal.emit(90)
        self.log_signal.emit("Normalizing UV coordinates...")
        
        # Normalize to [0,1] range
        uv_min = np.min(uv_vertices, axis=0)
        uv_max = np.max(uv_vertices, axis=0)
        uv_range = uv_max - uv_min
        if uv_range[0] > 0 and uv_range[1] > 0:
            uv_vertices = (uv_vertices - uv_min) / uv_range.max() * 0.9 + 0.05
        
        self.progress_signal.emit(100)
        return uv_vertices, self.faces

class MeshViewer3D(QOpenGLWidget):
    # ... (keep the existing MeshViewer3D class exactly as before) ...

class UVLayoutViewer(QOpenGLWidget):
    # ... (keep the existing UVLayoutViewer class exactly as before) ...

class ComparisonViewer(QWidget):
    # ... (keep the existing ComparisonViewer class exactly as before) ...

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UV Conformal Map Tool - Fabric Forming")
        self.setGeometry(100, 100, 1400, 900)
        
        self.vertices = None
        self.faces = None
        self.uv_vertices = None
        self.uv_faces = None
        
        self.mapping_thread = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create tabs for better organization
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Main visualization tab
        vis_tab = QWidget()
        vis_layout = QVBoxLayout(vis_tab)
        
        # File loading section
        file_group = QGroupBox("File Operations")
        file_layout = QHBoxLayout(file_group)
        
        self.load_btn = QPushButton("Load 3D Model")
        self.load_btn.clicked.connect(self.load_model)
        file_layout.addWidget(self.load_btn)
        
        self.process_btn = QPushButton("Compute UV Map")
        self.process_btn.clicked.connect(self.compute_uv_map)
        self.process_btn.setEnabled(False)
        file_layout.addWidget(self.process_btn)
        
        file_layout.addStretch()
        
        self.file_label = QLabel("No file loaded")
        file_layout.addWidget(self.file_label)
        
        vis_layout.addWidget(file_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        vis_layout.addWidget(self.progress_bar)
        
        # Add comparison viewer
        self.comparison_viewer = ComparisonViewer()
        vis_layout.addWidget(self.comparison_viewer)
        
        tabs.addTab(vis_tab, "Visualization")
        
        # Log tab
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(QLabel("Processing Log:"))
        log_layout.addWidget(self.log_text)
        
        tabs.addTab(log_tab, "Log")
        
        # Instructions tab
        instructions_tab = QWidget()
        instructions_layout = QVBoxLayout(instructions_tab)
        
        instructions = """
        <h3>UV Conformal Map Tool - Instructions</h3>
        
        <h4>Step 1: Load 3D Model</h4>
        <ul>
        <li>Click 'Load 3D Model' to import your 3D mesh</li>
        <li>Supported formats: OBJ, STL, PLY</li>
        <li>The 3D model will appear in the left viewer</li>
        </ul>
        
        <h4>Step 2: Compute UV Map</h4>
        <ul>
        <li>Click 'Compute UV Map' to generate a conformal UV mapping</li>
        <li>This may take a few seconds depending on mesh complexity</li>
        <li>Progress will be shown in the progress bar and log</li>
        </ul>
        
        <h4>Step 3: Visualize and Export</h4>
        <ul>
        <li>View the UV layout in the right panel</li>
        <li>Toggle between wireframe and distortion heatmap views</li>
        <li>Use 'Export Fabric Pattern' to create SVG for fabric forming</li>
        <li>Adjust scale and seam allowance as needed</li>
        </ul>
        
        <h4>Fabric Forming Process:</h4>
        <ol>
        <li>Print the exported SVG pattern at 100% scale</li>
        <li>Cut along the black lines</li>
        <li>Place on pre-stretched fabric</li>
        <li>Adhere pattern to fabric</li>
        <li>Release fabric tension to form 3D shape</li>
        </ol>
        """
        
        instructions_label = QLabel(instructions)
        instructions_label.setWordWrap(True)
        instructions_layout.addWidget(instructions_label)
        
        tabs.addTab(instructions_tab, "Instructions")
        
        self.log("Application started. Load a 3D model to begin.")
        
    def log(self, message):
        """Add message to log with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
    def load_model(self):
        """Load a 3D model file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open 3D Model", "", 
            "3D Files (*.obj *.stl *.ply);;All Files (*)"
        )
        
        if not filename:
            return
            
        try:
            self.log(f"Loading model: {os.path.basename(filename)}")
            self.file_label.setText(f"Loaded: {os.path.basename(filename)}")
            
            # Load mesh using trimesh
            mesh = trimesh.load_mesh(filename)
            self.vertices = mesh.vertices.astype(np.float32)
            self.faces = mesh.faces.astype(np.int32)
            
            self.log(f"Mesh loaded: {len(self.vertices)} vertices, {len(self.faces)} faces")
            
            # Update 3D viewer
            self.comparison_viewer.mesh_viewer.set_mesh(self.vertices, self.faces)
            
            # Enable process button
            self.process_btn.setEnabled(True)
            
            # Clear previous UV data
            self.uv_vertices = None
            self.uv_faces = None
            self.comparison_viewer.uv_viewer.set_uv_layout(None, None)
            
            self.log("Model loaded successfully. Ready for UV mapping.")
            
        except Exception as e:
            self.log(f"Error loading model: {str(e)}")
            QMessageBox.critical(self, "Load Error", f"Failed to load model:\n{str(e)}")
    
    def compute_uv_map(self):
        """Compute conformal UV mapping"""
        if self.vertices is None or self.faces is None:
            QMessageBox.warning(self, "No Model", "Please load a 3D model first.")
            return
            
        # Disable buttons during processing
        self.load_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start mapping thread
        self.mapping_thread = ConformalMappingThread(self.vertices, self.faces)
        self.mapping_thread.progress_signal.connect(self.progress_bar.setValue)
        self.mapping_thread.log_signal.connect(self.log)
        self.mapping_thread.finished_signal.connect(self.on_mapping_finished)
        self.mapping_thread.start()
        
        self.log("Starting UV mapping computation...")
    
    def on_mapping_finished(self, uv_vertices, uv_faces):
        """Handle completion of UV mapping"""
        # Re-enable buttons
        self.load_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if uv_vertices is not None and uv_faces is not None:
            self.uv_vertices = uv_vertices
            self.uv_faces = uv_faces
            
            # Update comparison viewer with both 3D and UV data
            self.comparison_viewer.set_mesh_data(
                self.vertices, self.faces, 
                self.uv_vertices, self.uv_faces
            )
            
            self.log("UV mapping completed. Ready for export.")
            
            # Switch to distortion view
            self.comparison_viewer.uv_mode_combo.setCurrentText("Conformal Distortion")
            
        else:
            self.log("UV mapping failed.")
            QMessageBox.warning(self, "Mapping Error", "Failed to compute UV mapping.")
    
    def closeEvent(self, event):
        """Handle application closure"""
        if self.mapping_thread and self.mapping_thread.isRunning():
            self.mapping_thread.terminate()
            self.mapping_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())
