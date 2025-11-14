import numpy as np
from PyQt5.QtWidgets import (QOpenGLWidget, QVBoxLayout, QHBoxLayout, QWidget, 
                             QLabel, QSplitter, QCheckBox)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QMouseEvent
from OpenGL.GL import *
from OpenGL.GLU import *

class MeshViewer3D(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vertices = None
        self.faces = None
        
        # Camera controls
        self.rotation_x = -45
        self.rotation_y = 45
        self.zoom = -3
        self.last_pos = QPoint()
        self.wireframe = False
        
        # Mesh bounds for auto-zoom
        self.mesh_center = np.array([0, 0, 0])
        self.mesh_radius = 1.0
        
    def set_mesh(self, vertices, faces):
        """Set mesh data for display"""
        self.vertices = vertices
        self.faces = faces
        
        # Calculate mesh bounds for auto-zoom
        if vertices is not None and len(vertices) > 0:
            self.mesh_center = np.mean(vertices, axis=0)
            # Calculate the radius as the maximum distance from center
            distances = np.linalg.norm(vertices - self.mesh_center, axis=1)
            self.mesh_radius = np.max(distances) if len(distances) > 0 else 1.0
            
            # Auto-adjust zoom to fit mesh
            self.zoom = -self.mesh_radius * 2.5  # Adjust multiplier as needed
            
        self.update()
        
    def initializeGL(self):
        """Initialize OpenGL"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set up lighting
        glLightfv(GL_LIGHT0, GL_POSITION, [2, 2, 2, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.7, 0.7, 0.7, 1])
        
        glClearColor(0.95, 0.95, 0.95, 1.0)
        
    def resizeGL(self, width, height):
        """Handle window resize"""
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = width / float(height) if height != 0 else 1.0
        gluPerspective(45, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
    def paintGL(self):
        """Render the scene"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Camera position
        glTranslatef(0, 0, self.zoom)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        
        # Center the mesh
        if self.vertices is not None and len(self.vertices) > 0:
            glTranslatef(-self.mesh_center[0], -self.mesh_center[1], -self.mesh_center[2])
        
        if self.vertices is not None and self.faces is not None:
            self.draw_mesh()
                
    def draw_mesh(self):
        """Draw the 3D mesh"""
        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDisable(GL_LIGHTING)
            glColor3f(0.2, 0.2, 0.8)
            glLineWidth(1.5)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glEnable(GL_LIGHTING)
            glColor3f(0.6, 0.7, 0.9)
            
        glBegin(GL_TRIANGLES)
        for face in self.faces:
            if len(face) == 3:  # Triangle
                for vertex_idx in face:
                    if vertex_idx < len(self.vertices):
                        glVertex3fv(self.vertices[vertex_idx])
        glEnd()
        
        # Reset to defaults
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_LIGHTING)
        glLineWidth(1.0)
        
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for rotation"""
        self.last_pos = event.pos()
        
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse drag for rotation"""
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        
        if event.buttons() & Qt.LeftButton:
            self.rotation_y += dx * 0.5
            self.rotation_x += dy * 0.5
            self.update()
            
        self.last_pos = event.pos()
        
    def wheelEvent(self, event):
        """Handle mouse wheel for zoom"""
        zoom_speed = max(0.1, abs(self.zoom) * 0.05)  # Adaptive zoom speed
        self.zoom += event.angleDelta().y() * 0.001 * zoom_speed
        
        # Much wider zoom limits
        self.zoom = max(-50.0, min(-0.1, self.zoom))
        self.update()
        
    def set_wireframe(self, wireframe):
        """Set wireframe mode"""
        self.wireframe = wireframe
        self.update()

class UVLayoutViewer(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.uv_vertices = None
        self.uv_faces = None
        self.wireframe = False
        
        # UV bounds for proper scaling
        self.uv_center = np.array([0.5, 0.5])
        self.uv_scale = 1.0
        
    def set_uv_layout(self, uv_vertices, uv_faces):
        """Set UV layout data for display"""
        self.uv_vertices = uv_vertices
        self.uv_faces = uv_faces
        
        # Calculate UV bounds for proper display
        if uv_vertices is not None and len(uv_vertices) > 0:
            uv_array = np.array(uv_vertices)
            if len(uv_array) > 0:
                min_uv = np.min(uv_array, axis=0)
                max_uv = np.max(uv_array, axis=0)
                self.uv_center = (min_uv + max_uv) / 2.0
                self.uv_scale = max(np.max(max_uv - min_uv) * 0.6, 0.1)  # Scale to fit
                
        self.update()
        
    def initializeGL(self):
        """Initialize OpenGL"""
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        
    def resizeGL(self, width, height):
        """Handle window resize"""
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        # Orthographic projection for 2D UV space
        aspect = width / float(height) if height != 0 else 1.0
        
        # Use calculated scale, but ensure we can see the entire [0,1] range
        scale = max(self.uv_scale, 1.0) * 1.2  # Add some padding
        
        if aspect > 1:
            glOrtho(-scale, scale, -scale/aspect, scale/aspect, -1, 1)
        else:
            glOrtho(-scale*aspect, scale*aspect, -scale, scale, -1, 1)
            
        glMatrixMode(GL_MODELVIEW)
        
    def paintGL(self):
        """Render the UV layout"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Center the UV layout
        if self.uv_vertices is not None and len(self.uv_vertices) > 0:
            glTranslatef(-self.uv_center[0], -self.uv_center[1], 0)
        
        if self.uv_vertices is not None and self.uv_faces is not None:
            self.draw_uv_layout()
            self.draw_uv_boundary()
                
    def draw_uv_layout(self):
        """Draw the UV layout with proper coloring"""
        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glColor3f(0.8, 0.2, 0.2)
            glLineWidth(1.5)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            
            # Use different colors for different faces to see the structure
            glColor3f(0.9, 0.8, 0.6)  # Default color
            
        glBegin(GL_TRIANGLES)
        for i, face in enumerate(self.uv_faces):
            if len(face) == 3:  # Triangle
                # Alternate colors for different faces in filled mode
                if not self.wireframe:
                    # Create a color based on face index to differentiate faces
                    hue = (i * 0.6180339887) % 1.0  # Golden ratio for distribution
                    r = 0.6 + 0.3 * (hue % 0.333)
                    g = 0.6 + 0.3 * ((hue + 0.333) % 0.333)
                    b = 0.6 + 0.3 * ((hue + 0.666) % 0.333)
                    glColor3f(r, g, b)
                
                for uv_idx in face:
                    if uv_idx < len(self.uv_vertices):
                        uv = self.uv_vertices[uv_idx]
                        glVertex3f(uv[0], uv[1], 0)
        glEnd()
        
        # Reset to defaults
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glLineWidth(1.0)
        
    def draw_uv_boundary(self):
        """Draw UV space boundary and grid"""
        # Draw the [0,1] boundary
        glColor3f(0.8, 0.8, 0.8)
        glBegin(GL_LINE_LOOP)
        glVertex3f(0, 0, 0)
        glVertex3f(1, 0, 0)
        glVertex3f(1, 1, 0)
        glVertex3f(0, 1, 0)
        glEnd()
        
        # Draw grid lines for better orientation
        glColor3f(0.9, 0.9, 0.9)
        glBegin(GL_LINES)
        # Vertical lines
        for i in range(1, 4):
            x = i * 0.25
            glVertex3f(x, 0, 0)
            glVertex3f(x, 1, 0)
        # Horizontal lines
        for i in range(1, 4):
            y = i * 0.25
            glVertex3f(0, y, 0)
            glVertex3f(1, y, 0)
        glEnd()
        
    def set_wireframe(self, wireframe):
        """Set wireframe mode"""
        self.wireframe = wireframe
        self.update()

class ComparisonViewer(QWidget):
    """Widget that shows 3D mesh and UV layout side by side"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)  # Minimize margins
        layout.setSpacing(4)  # Minimize spacing
        
        # Splitter for side-by-side view - use maximum space
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - 3D mesh with minimal label
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(2, 2, 2, 2)
        left_layout.setSpacing(2)
        
        left_label = QLabel("3D Mesh • Drag to rotate • Wheel to zoom")
        left_label.setMaximumHeight(15)  # Very small label
        left_label.setAlignment(Qt.AlignCenter)
        left_label.setStyleSheet("color: gray; font-size: 9px;")
        left_layout.addWidget(left_label)
        
        self.mesh_viewer = MeshViewer3D()
        left_layout.addWidget(self.mesh_viewer)
        splitter.addWidget(left_widget)
        
        # Right side - UV layout with minimal label
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(2, 2, 2, 2)
        right_layout.setSpacing(2)
        
        right_label = QLabel("UV Layout • Different colors = different faces")
        right_label.setMaximumHeight(15)  # Very small label
        right_label.setAlignment(Qt.AlignCenter)
        right_label.setStyleSheet("color: gray; font-size: 9px;")
        right_layout.addWidget(right_label)
        
        self.uv_viewer = UVLayoutViewer()
        right_layout.addWidget(self.uv_viewer)
        splitter.addWidget(right_widget)
        
        # Set equal sizes and make splitter stretch
        splitter.setSizes([500, 500])
        layout.addWidget(splitter, 1)  # Give splitter stretch factor
        
        # Minimal controls at the bottom
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(2, 2, 2, 2)
        
        self.wireframe_checkbox = QCheckBox("Wireframe")
        self.wireframe_checkbox.stateChanged.connect(self.on_wireframe_changed)
        controls_layout.addWidget(self.wireframe_checkbox)
        
        controls_layout.addStretch()
        
        # Reset view button
        self.reset_view_btn = QPushButton("Reset Views")
        self.reset_view_btn.clicked.connect(self.reset_views)
        self.reset_view_btn.setMaximumWidth(80)
        controls_layout.addWidget(self.reset_view_btn)
        
        layout.addLayout(controls_layout)
        
    def set_mesh_data(self, vertices, faces, uv_vertices, uv_faces):
        """Set mesh data for both viewers"""
        self.mesh_viewer.set_mesh(vertices, faces)
        self.uv_viewer.set_uv_layout(uv_vertices, uv_faces)
        
    def on_wireframe_changed(self, state):
        """Handle wireframe toggle for both viewers"""
        wireframe = (state == Qt.Checked)
        self.mesh_viewer.set_wireframe(wireframe)
        self.uv_viewer.set_wireframe(wireframe)
        
    def reset_views(self):
        """Reset both views to default"""
        # For 3D view, we need to recreate the mesh to reset auto-zoom
        if hasattr(self.mesh_viewer, 'vertices') and self.mesh_viewer.vertices is not None:
            self.mesh_viewer.set_mesh(self.mesh_viewer.vertices, self.mesh_viewer.faces)
        
        # For UV view, reset rotation and recreate
        if hasattr(self.uv_viewer, 'uv_vertices') and self.uv_viewer.uv_vertices is not None:
            self.uv_viewer.set_uv_layout(self.uv_viewer.uv_vertices, self.uv_viewer.uv_faces)
