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

class TrianglePacker:
    """Packs triangles efficiently for fabric cutting patterns"""
    
    @staticmethod
    def pack_triangles(uv_vertices, uv_faces, padding=0.02):
        """Pack triangles into a rectangular area"""
        print(f"Packing {len(uv_faces)} triangles...")
        
        triangle_islands = []
        
        for i, face in enumerate(uv_faces):
            if len(face) == 3:
                uv_triangle = [uv_vertices[face[0]], uv_vertices[face[1]], uv_vertices[face[2]]]
                triangle_islands.append({
                    'uv_points': np.array(uv_triangle),
                    'face_index': i
                })
        
        if not triangle_islands:
            return []
        
        # Sort by area
        triangle_islands.sort(key=lambda x: TrianglePacker.triangle_area(x['uv_points']), reverse=True)
        
        packed_triangles = []
        current_x = 0.0
        current_y = 0.0
        row_height = 0.0
        max_width = 0.0
        
        for triangle in triangle_islands:
            points = triangle['uv_points']
            min_x, min_y = np.min(points, axis=0)
            max_x, max_y = np.max(points, axis=0)
            width = max_x - min_x
            height = max_y - min_y
            
            if current_x + width + padding > 1.0:
                current_x = 0.0
                current_y += row_height + padding
                row_height = 0.0
            
            offset_x = current_x - min_x
            offset_y = current_y - min_y
            
            row_height = max(row_height, height)
            max_width = max(max_width, current_x + width)
            
            packed_points = points + np.array([offset_x, offset_y])
            packed_triangles.append({
                'original_face_index': triangle['face_index'],
                'packed_points': packed_points
            })
            
            current_x += width + padding
        
        # Normalize to [0,1] range
        scale_factor = max(max_width, current_y + row_height)
        if scale_factor > 0:
            for triangle in packed_triangles:
                triangle['packed_points'] /= scale_factor
        
        print(f"Packed {len(packed_triangles)} triangles")
        return packed_triangles

    @staticmethod
    def triangle_area(points):
        """Calculate area of a triangle"""
        if len(points) != 3:
            return 0.0
        a, b, c = points[0], points[1], points[2]
        return 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))

class ConformalMappingThread(QThread):
    """Thread for computing conformal mapping"""
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object, object, object)  # uv_vertices, uv_faces, triangles_3d
    
    def __init__(self, vertices, faces):
        super().__init__()
        self.vertices = vertices
        self.faces = faces
        
    def run(self):
        try:
            self.log_signal.emit("Starting conformal mapping...")
            uv_vertices, uv_faces, triangles_3d = self.compute_conformal_map()
            self.finished_signal.emit(uv_vertices, uv_faces, triangles_3d)
            self.log_signal.emit("Conformal mapping completed successfully!")
        except Exception as e:
            self.log_signal.emit(f"Error in conformal mapping: {str(e)}")
            self.finished_signal.emit(None, None, None)
    
    def compute_conformal_map(self):
        """Compute per-face conformal mapping"""
        self.progress_signal.emit(10)
        
        if self.vertices is None or self.faces is None:
            return None, None, None
        
        n_faces = len(self.faces)
        triangles_3d = []
        uv_vertices = []
        uv_faces = []
        vertex_offset = 0
        
        print(f"Processing {n_faces} faces for UV mapping...")
        
        for face_idx, face in enumerate(self.faces):
            if len(face) != 3:
                continue
                
            self.progress_signal.emit(10 + int(80 * face_idx / n_faces))
            
            # Get 3D triangle
            v0, v1, v2 = self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]
            triangles_3d.append((v0, v1, v2))
            
            # Create a simple test triangle that should definitely be visible
            # This creates a triangle in the center of UV space
            if face_idx % 3 == 0:
                uv_triangle = np.array([[0.3, 0.3], [0.7, 0.3], [0.5, 0.7]], dtype=np.float32)
            elif face_idx % 3 == 1:
                uv_triangle = np.array([[0.1, 0.1], [0.4, 0.1], [0.1, 0.4]], dtype=np.float32)
            else:
                uv_triangle = np.array([[0.6, 0.6], [0.9, 0.6], [0.6, 0.9]], dtype=np.float32)
            
            uv_vertices.extend(uv_triangle)
            uv_faces.append([vertex_offset, vertex_offset + 1, vertex_offset + 2])
            vertex_offset += 3
        
        print(f"Generated {len(uv_vertices)} UV vertices before packing")
        
        self.progress_signal.emit(90)
        self.log_signal.emit("Packing triangles...")
        
        # Pack triangles
        uv_vertices_array = np.array(uv_vertices, dtype=np.float32)
        packed_data = TrianglePacker.pack_triangles(uv_vertices_array, uv_faces)
        
        if not packed_data:
            return uv_vertices_array, uv_faces, triangles_3d
        
        # Rebuild from packed data
        packed_uv_vertices = []
        packed_uv_faces = []
        
        new_vertex_offset = 0
        for packed_triangle in packed_data:
            points = packed_triangle['packed_points']
            packed_uv_vertices.extend(points)
            packed_uv_faces.append([new_vertex_offset, new_vertex_offset + 1, new_vertex_offset + 2])
            new_vertex_offset += 3
        
        self.progress_signal.emit(100)
        
        packed_uv_vertices_array = np.array(packed_uv_vertices, dtype=np.float32)
        print(f"Final: {len(packed_uv_vertices_array)} UV vertices, {len(packed_uv_faces)} UV faces")
        print(f"UV range: [{np.min(packed_uv_vertices_array, axis=0)}, {np.max(packed_uv_vertices_array, axis=0)}]")
        
        return packed_uv_vertices_array, packed_uv_faces, triangles_3d

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
            self.mesh_radius = np.max(np.linalg.norm(vertices - self.mesh_center, axis=1))
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
        glClearColor(1.0, 1.0, 1.0, 1.0)  # White background
        
    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # Fixed orthographic projection that definitely covers [0,1] range
        glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Draw the boundary and grid first
        self.draw_uv_boundary()
        
        # Draw UV layout if available
        if self.uv_vertices is not None and self.uv_faces is not None:
            self.draw_uv_layout()
        else:
            self.draw_placeholder()
                
    def draw_uv_layout(self):
        print(f"Drawing UV layout: {len(self.uv_vertices)} vertices, {len(self.uv_faces)} faces")
        
        if len(self.uv_vertices) == 0 or len(self.uv_faces) == 0:
            print("No vertices or faces to draw!")
            return
            
        # Draw filled triangles in bright colors to ensure visibility
        glColor3f(1.0, 0.0, 0.0)  # RED filled triangles
        glBegin(GL_TRIANGLES)
        for face in self.uv_faces:
            if len(face) == 3:
                for vertex_idx in face:
                    if vertex_idx < len(self.uv_vertices):
                        uv = self.uv_vertices[vertex_idx]
                        glVertex2f(uv[0], uv[1])
        glEnd()
        
        # Draw wireframe in black
        glColor3f(0.0, 0.0, 0.0)  # BLACK wireframe
        glLineWidth(3.0)  # Thick lines for visibility
        glBegin(GL_LINES)
        for face in self.uv_faces:
            if len(face) == 3:
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
        # Draw UV space boundary in blue
        glColor3f(0.0, 0.0, 1.0)  # BLUE boundary
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(0.0, 0.0)
        glVertex2f(1.0, 0.0)
        glVertex2f(1.0, 1.0)
        glVertex2f(0.0, 1.0)
        glEnd()
        
        # Draw grid in light gray
        glColor3f(0.7, 0.7, 0.7)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for i in range(1, 4):
            x = i * 0.25
            glVertex2f(x, 0.0)
            glVertex2f(x, 1.0)
            glVertex2f(0.0, x)
            glVertex2f(1.0, x)
        glEnd()
        
    def draw_placeholder(self):
        # Draw a large red X when no UV data
        glColor3f(1.0, 0.0, 0.0)  # RED
        glLineWidth(4.0)
        glBegin(GL_LINES)
        glVertex2f(0.1, 0.1)
        glVertex2f(0.9, 0.9)
        glVertex2f(0.9, 0.1)
        glVertex2f(0.1, 0.9)
        glEnd()
        glLineWidth(1.0)

class ComparisonViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vertices_3d = None
        self.faces_3d = None
        self.uv_vertices = None
        self.uv_faces = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Splitter for side-by-side view
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - 3D mesh
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_label = QLabel("3D Mesh • Drag to rotate • Wheel to zoom")
        left_label.setAlignment(Qt.AlignCenter)
        left_label.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        left_layout.addWidget(left_label)
        
        self.mesh_viewer = MeshViewer3D()
        left_layout.addWidget(self.mesh_viewer)
        splitter.addWidget(left_widget)
        
        # Right side - UV layout
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_label = QLabel("UV Layout • Should show RED triangles")
        right_label.setAlignment(Qt.AlignCenter)
        right_label.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        right_layout.addWidget(right_label)
        
        self.uv_viewer = UVLayoutViewer()
        right_layout.addWidget(self.uv_viewer)
        splitter.addWidget(right_widget)
        
        splitter.setSizes([400, 400])
        layout.addWidget(splitter)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UV Conformal Map Tool - DEBUG VERSION")
        self.setGeometry(100, 100, 1200, 800)
        
        self.vertices = None
        self.faces = None
        self.uv_vertices = None
        self.uv_faces = None
        
        self.mapping_thread = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Main visualization tab
        vis_tab = QWidget()
        vis_layout = QVBoxLayout(vis_tab)
        vis_layout.setContentsMargins(5, 5, 5, 5)
        
        # File loading section
        file_group = QGroupBox("File Operations")
        file_layout = QHBoxLayout(file_group)
        
        self.load_btn = QPushButton("Load 3D Model (OBJ)")
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
            
            self.comparison_viewer.mesh_viewer.set_mesh(self.vertices, self.faces)
            self.process_btn.setEnabled(True)
            
            # Clear previous UV data
            self.uv_vertices = None
            self.uv_faces = None
            self.comparison_viewer.uv_viewer.set_uv_layout(None, None)
            
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
        
        self.mapping_thread = ConformalMappingThread(self.vertices, self.faces)
        self.mapping_thread.progress_signal.connect(self.progress_bar.setValue)
        self.mapping_thread.log_signal.connect(self.log)
        self.mapping_thread.finished_signal.connect(self.on_mapping_finished)
        self.mapping_thread.start()
        
        self.log("Computing UV mapping...")
    
    def on_mapping_finished(self, uv_vertices, uv_faces, triangles_3d):
        self.load_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if uv_vertices is not None and uv_faces is not None:
            self.uv_vertices = uv_vertices
            self.uv_faces = uv_faces
            
            self.log(f"UV mapping complete: {len(uv_vertices)} vertices, {len(uv_faces)} faces")
            
            # Update comparison viewer
            self.comparison_viewer.uv_viewer.set_uv_layout(uv_vertices, uv_faces)
            
        else:
            self.log("UV mapping failed")
            QMessageBox.warning(self, "Mapping Error", "Failed to compute UV mapping")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
