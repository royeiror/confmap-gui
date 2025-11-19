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
import sys
import time

class SimpleObjLoader:
    """Simple OBJ file loader"""
    
    @staticmethod
    def load_obj(filename):
        """Load vertices and faces from OBJ file"""
        vertices = []
        faces = []
        
        try:
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if not parts:
                        continue
                    
                    if parts[0] == 'v':  # vertex
                        if len(parts) >= 4:
                            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    
                    elif parts[0] == 'f':  # face
                        face_vertices = []
                        for part in parts[1:]:
                            vertex_data = part.split('/')[0]
                            if vertex_data:
                                try:
                                    vertex_idx = int(vertex_data) - 1
                                    if vertex_idx >= 0:
                                        face_vertices.append(vertex_idx)
                                except ValueError:
                                    continue
                        
                        if len(face_vertices) >= 3:
                            for i in range(1, len(face_vertices) - 1):
                                faces.append([face_vertices[0], face_vertices[i], face_vertices[i + 1]])
            
            return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)
            
        except Exception as e:
            raise ValueError(f"Error parsing OBJ file: {str(e)}")

class UVMappingThread(QThread):
    """Thread for computing UV mapping"""
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object, object)  # uv_vertices, uv_faces
    
    def __init__(self, vertices, faces):
        super().__init__()
        self.vertices = vertices
        self.faces = faces
        
    def run(self):
        try:
            self.log_signal.emit("Starting UV mapping...")
            uv_vertices, uv_faces = self.compute_simple_uv_map()
            self.finished_signal.emit(uv_vertices, uv_faces)
            self.log_signal.emit("UV mapping completed successfully!")
        except Exception as e:
            self.log_signal.emit(f"Error in UV mapping: {str(e)}")
            self.finished_signal.emit(None, None)
    
    def compute_simple_uv_map(self):
        """Compute a simple planar UV mapping"""
        self.progress_signal.emit(10)
        
        if self.vertices is None or self.faces is None:
            return None, None
        
        n_faces = len(self.faces)
        uv_vertices = []
        uv_faces = []
        vertex_offset = 0
        
        print(f"Processing {n_faces} faces for UV mapping...")
        
        for face_idx, face in enumerate(self.faces):
            if len(face) != 3:
                continue
                
            self.progress_signal.emit(10 + int(80 * face_idx / n_faces))
            
            # Get 3D triangle vertices
            v0, v1, v2 = self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]
            
            # Simple planar projection - just use XZ coordinates normalized to [0,1]
            # This creates a basic UV mapping for testing
            points_3d = np.array([v0, v1, v2])
            
            # Get bounding box in XZ plane
            min_coords = np.min(points_3d[:, [0, 2]], axis=0)
            max_coords = np.max(points_3d[:, [0, 2]], axis=0)
            
            # Normalize to [0,1] range
            if np.any(max_coords - min_coords > 1e-10):
                normalized_points = (points_3d[:, [0, 2]] - min_coords) / (max_coords - min_coords)
            else:
                normalized_points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
            
            # Add some offset to separate triangles
            group_offset = (face_idx % 10) * 0.1
            normalized_points += group_offset
            
            # Ensure we stay in [0,1] range
            normalized_points = np.clip(normalized_points, 0, 1)
            
            # Add to UV arrays
            uv_vertices.extend(normalized_points)
            uv_faces.append([vertex_offset, vertex_offset + 1, vertex_offset + 2])
            vertex_offset += 3
        
        self.progress_signal.emit(100)
        
        uv_vertices_array = np.array(uv_vertices, dtype=np.float32)
        print(f"Generated {len(uv_vertices_array)} UV vertices, {len(uv_faces)} UV faces")
        print(f"UV range: [{np.min(uv_vertices_array, axis=0)}, {np.max(uv_vertices_array, axis=0)}]")
        
        return uv_vertices_array, uv_faces

class MeshViewer3D(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vertices = None
        self.faces = None
        self.rotation_x = -45
        self.rotation_y = 45
        self.zoom = -5
        self.last_pos = QPoint()
        
    def set_mesh(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        if vertices is not None and len(vertices) > 0:
            self.mesh_center = np.mean(vertices, axis=0)
            distances = np.linalg.norm(vertices - self.mesh_center, axis=1)
            self.mesh_radius = np.max(distances) if len(distances) > 0 else 1.0
            self.zoom = -self.mesh_radius * 3.0
        self.update()
        
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glLightfv(GL_LIGHT0, GL_POSITION, [2, 2, 2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])
        
    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = width / float(height) if height != 0 else 1.0
        gluPerspective(45, aspect, 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)
        
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        glTranslatef(0, 0, self.zoom)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        
        if self.vertices is not None and len(self.vertices) > 0:
            glTranslatef(-self.mesh_center[0], -self.mesh_center[1], -self.mesh_center[2])
        
        if self.vertices is not None and self.faces is not None:
            self.draw_mesh()
                
    def draw_mesh(self):
        glColor3f(0.6, 0.7, 0.9)
        glBegin(GL_TRIANGLES)
        for face in self.faces:
            if len(face) == 3:
                for vertex_idx in face:
                    glVertex3fv(self.vertices[vertex_idx])
        glEnd()
        
    def mousePressEvent(self, event: QMouseEvent):
        self.last_pos = event.pos()
        
    def mouseMoveEvent(self, event: QMouseEvent):
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        
        if event.buttons() & Qt.LeftButton:
            self.rotation_y += dx * 0.5
            self.rotation_x += dy * 0.5
            self.update()
            
        self.last_pos = event.pos()
        
    def wheelEvent(self, event):
        self.zoom += event.angleDelta().y() * 0.001
        self.update()

class UVLayoutViewer(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.uv_vertices = None
        self.uv_faces = None
        
    def set_uv_layout(self, uv_vertices, uv_faces):
        print(f"UVLayoutViewer: Setting UV layout with {len(uv_vertices) if uv_vertices is not None else 0} vertices, "
              f"{len(uv_faces) if uv_faces is not None else 0} faces")
        self.uv_vertices = uv_vertices
        self.uv_faces = uv_faces
        self.update()
        
    def initializeGL(self):
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        
    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # Simple orthographic projection for 2D UV space
        glOrtho(-0.1, 1.1, -0.1, 1.1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Always draw the boundary
        self.draw_uv_boundary()
        
        # Draw UV layout if available
        if self.uv_vertices is not None and self.uv_faces is not None:
            self.draw_uv_layout()
        else:
            self.draw_placeholder()
                
    def draw_uv_layout(self):
        print(f"Drawing UV layout: {len(self.uv_vertices)} vertices, {len(self.uv_faces)} faces")
        
        # Draw filled triangles in light blue
        glColor3f(0.7, 0.8, 1.0)
        glBegin(GL_TRIANGLES)
        for face in self.uv_faces:
            if len(face) == 3:
                for vertex_idx in face:
                    if vertex_idx < len(self.uv_vertices):
                        uv = self.uv_vertices[vertex_idx]
                        glVertex2f(uv[0], uv[1])
        glEnd()
        
        # Draw wireframe in dark blue
        glColor3f(0.2, 0.2, 0.8)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        for face in self.uv_faces:
            if len(face) == 3:
                # Draw three edges for each triangle
                for i in range(3):
                    v1 = face[i]
                    v2 = face[(i + 1) % 3]
                    if v1 < len(self.uv_vertices) and v2 < len(self.uv_vertices):
                        uv1 = self.uv_vertices[v1]
                        uv2 = self.uv_vertices[v2]
                        glVertex2f(uv1[0], uv1[1])
                        glVertex2f(uv2[0], uv2[1])
        glEnd()
        glLineWidth(1.0)
        
    def draw_uv_boundary(self):
        # Draw UV space boundary
        glColor3f(0.5, 0.5, 0.5)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(0, 0)
        glVertex2f(1, 0)
        glVertex2f(1, 1)
        glVertex2f(0, 1)
        glEnd()
        
        # Draw grid
        glColor3f(0.8, 0.8, 0.8)
        glLineWidth(0.5)
        glBegin(GL_LINES)
        for i in range(1, 4):
            x = i * 0.25
            glVertex2f(x, 0)
            glVertex2f(x, 1)
            glVertex2f(0, x)
            glVertex2f(1, x)
        glEnd()
        glLineWidth(1.0)
        
    def draw_placeholder(self):
        # Draw a placeholder when no UV data is available
        glColor3f(0.8, 0.8, 0.8)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex2f(0.1, 0.1)
        glVertex2f(0.9, 0.9)
        glVertex2f(0.9, 0.1)
        glVertex2f(0.1, 0.9)
        glEnd()
        glLineWidth(1.0)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UV Map Tool - Simple Test")
        self.setGeometry(100, 100, 1200, 800)
        
        self.vertices = None
        self.faces = None
        self.uv_vertices = None
        self.uv_faces = None
        
        self.mapping_thread = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - 3D viewer
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_label = QLabel("3D Mesh Viewer")
        left_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(left_label)
        
        self.mesh_viewer = MeshViewer3D()
        left_layout.addWidget(self.mesh_viewer)
        splitter.addWidget(left_widget)
        
        # Right side - UV viewer
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_label = QLabel("UV Layout Viewer")
        right_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(right_label)
        
        self.uv_viewer = UVLayoutViewer()
        right_layout.addWidget(self.uv_viewer)
        splitter.addWidget(right_widget)
        
        splitter.setSizes([600, 600])
        layout.addWidget(splitter)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load OBJ File")
        self.load_btn.clicked.connect(self.load_model)
        controls_layout.addWidget(self.load_btn)
        
        self.process_btn = QPushButton("Generate UV Map")
        self.process_btn.clicked.connect(self.compute_uv_map)
        self.process_btn.setEnabled(False)
        controls_layout.addWidget(self.process_btn)
        
        controls_layout.addStretch()
        
        self.file_label = QLabel("No file loaded")
        controls_layout.addWidget(self.file_label)
        
        layout.addLayout(controls_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Log
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        self.log_text.setReadOnly(True)
        layout.addWidget(QLabel("Log:"))
        layout.addWidget(self.log_text)
        
        self.log("Application started. Load an OBJ file to begin.")
        
    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
    def load_model(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open 3D Model", "", "OBJ Files (*.obj)")
            
        if not filename:
            return
            
        try:
            self.log(f"Loading: {os.path.basename(filename)}")
            self.vertices, self.faces = SimpleObjLoader.load_obj(filename)
            
            self.log(f"Loaded: {len(self.vertices)} vertices, {len(self.faces)} faces")
            self.file_label.setText(f"Loaded: {os.path.basename(filename)}")
            
            self.mesh_viewer.set_mesh(self.vertices, self.faces)
            self.process_btn.setEnabled(True)
            
            # Clear previous UV data
            self.uv_viewer.set_uv_layout(None, None)
            
        except Exception as e:
            self.log(f"Error: {str(e)}")
            QMessageBox.critical(self, "Load Error", f"Failed to load model:\n{str(e)}")
    
    def compute_uv_map(self):
        if self.vertices is None or self.faces is None:
            return
            
        self.load_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.mapping_thread = UVMappingThread(self.vertices, self.faces)
        self.mapping_thread.progress_signal.connect(self.progress_bar.setValue)
        self.mapping_thread.log_signal.connect(self.log)
        self.mapping_thread.finished_signal.connect(self.on_mapping_finished)
        self.mapping_thread.start()
        
        self.log("Computing UV mapping...")
    
    def on_mapping_finished(self, uv_vertices, uv_faces):
        self.load_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if uv_vertices is not None and uv_faces is not None:
            self.uv_vertices = uv_vertices
            self.uv_faces = uv_faces
            self.uv_viewer.set_uv_layout(uv_vertices, uv_faces)
            self.log(f"UV mapping complete: {len(uv_vertices)} vertices, {len(uv_faces)} faces")
        else:
            self.log("UV mapping failed")
            QMessageBox.warning(self, "Mapping Error", "Failed to compute UV mapping")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
