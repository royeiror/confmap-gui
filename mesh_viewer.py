import numpy as np
from PyQt5.QtWidgets import (QOpenGLWidget, QVBoxLayout, QHBoxLayout, QWidget, 
                             QLabel, QSplitter, QCheckBox, QPushButton, QComboBox)
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
        self.zoom = -5
        self.last_pos = QPoint()
        self.wireframe_overlay = False
        
        # Mesh bounds for auto-zoom
        self.mesh_center = np.array([0, 0, 0])
        self.mesh_radius = 1.0
        
        # Projection settings
        self.near_plane = 0.1
        self.far_plane = 1000.0
        
    def set_mesh(self, vertices, faces):
        """Set mesh data for display"""
        self.vertices = vertices
        self.faces = faces
        
        # Calculate mesh bounds for auto-zoom
        if vertices is not None and len(vertices) > 0:
            self.mesh_center = np.mean(vertices, axis=0)
            distances = np.linalg.norm(vertices - self.mesh_center, axis=1)
            self.mesh_radius = np.max(distances) if len(distances) > 0 else 1.0
            self.zoom = -self.mesh_radius * 3.0
            
        self.update()
        
    def initializeGL(self):
        """Initialize OpenGL"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set up main lighting
        glLightfv(GL_LIGHT0, GL_POSITION, [2, 2, 2, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.5, 0.5, 0.5, 1])
        
        # Set up secondary lighting
        glLightfv(GL_LIGHT1, GL_POSITION, [-2, -1, -1, 1])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.3, 0.3, 0.3, 1])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.2, 0.2, 0.2, 1])
        
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.5, 0.5, 0.5, 1])
        glMaterialf(GL_FRONT, GL_SHININESS, 50.0)
        
        glClearColor(0.1, 0.1, 0.1, 1.0)
        
    def resizeGL(self, width, height):
        """Handle window resize"""
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = width / float(height) if height != 0 else 1.0
        gluPerspective(45, aspect, self.near_plane, self.far_plane)
        glMatrixMode(GL_MODELVIEW)
        
    def paintGL(self):
        """Render the scene"""
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
        """Draw the 3D mesh with optional wireframe overlay"""
        # Draw solid mesh first
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_LIGHTING)
        glColor3f(0.6, 0.7, 0.9)
        
        glBegin(GL_TRIANGLES)
        for face in self.faces:
            if len(face) == 3:
                v0 = self.vertices[face[0]]
                v1 = self.vertices[face[1]]
                v2 = self.vertices[face[2]]
                
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    normal = normal / norm
                    glNormal3fv(normal)
                
                for vertex_idx in face:
                    glVertex3fv(self.vertices[vertex_idx])
        glEnd()
        
        # Draw wireframe overlay if enabled
        if self.wireframe_overlay:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDisable(GL_LIGHTING)
            glColor3f(0.1, 0.1, 0.3)
            glLineWidth(1.0)
            
            glBegin(GL_TRIANGLES)
            for face in self.faces:
                if len(face) == 3:
                    for vertex_idx in face:
                        glVertex3fv(self.vertices[vertex_idx])
            glEnd()
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_LIGHTING)
        glLineWidth(1.0)
        
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
        zoom_speed = max(0.1, abs(self.zoom) * 0.05)
        self.zoom += event.angleDelta().y() * 0.001 * zoom_speed
        self.update()
        
    def set_wireframe_overlay(self, wireframe):
        self.wireframe_overlay = wireframe
        self.update()

class UVLayoutViewer(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.uv_vertices = None
        self.uv_faces = None
        self.display_mode = "wireframe"  # "wireframe", "heatmap"
        self.distortion = None  # Conformal distortion values per face
        self.distortion_range = (0.0, 1.0)  # Auto-adjusting range
        self.show_wireframe_overlay = True  # Optional wireframe in heatmap mode
        
    def set_uv_layout(self, uv_vertices, uv_faces, vertices_3d=None, faces_3d=None):
        """Set UV layout data and compute distortion if 3D data is provided"""
        self.uv_vertices = uv_vertices
        self.uv_faces = uv_faces
        
        # Compute conformal distortion if 3D data is available
        if vertices_3d is not None and faces_3d is not None and len(uv_vertices) > 0:
            self.compute_conformal_distortion(vertices_3d, faces_3d, uv_vertices, uv_faces)
        
        self.update()
    
    def compute_conformal_distortion(self, vertices_3d, faces_3d, uv_vertices, uv_faces):
        """Compute conformal distortion for each face using area ratio method"""
        self.distortion = []
        
        for i, face in enumerate(faces_3d):
            if len(face) != 3:
                self.distortion.append(0.0)  # Neutral for non-triangles
                continue
                
            try:
                # Get 3D triangle vertices
                v0_3d, v1_3d, v2_3d = vertices_3d[face[0]], vertices_3d[face[1]], vertices_3d[face[2]]
                
                # Get corresponding UV triangle vertices
                if i < len(uv_faces):
                    uv_face = uv_faces[i]
                else:
                    uv_face = face
                    
                # Ensure we have valid UV indices
                if (uv_face[0] >= len(uv_vertices) or uv_face[1] >= len(uv_vertices) or 
                    uv_face[2] >= len(uv_vertices)):
                    self.distortion.append(0.0)
                    continue
                    
                uv0, uv1, uv2 = uv_vertices[uv_face[0]], uv_vertices[uv_face[1]], uv_vertices[uv_face[2]]
                
                # Compute 3D area
                edge1_3d = v1_3d - v0_3d
                edge2_3d = v2_3d - v0_3d
                area_3d = 0.5 * np.linalg.norm(np.cross(edge1_3d, edge2_3d))
                
                # Compute UV area (in 2D)
                edge1_uv = uv1 - uv0
                edge2_uv = uv2 - uv0
                area_uv = 0.5 * abs(edge1_uv[0]*edge2_uv[1] - edge1_uv[1]*edge2_uv[0])
                
                # Avoid division by zero
                if area_3d < 1e-10 or area_uv < 1e-10:
                    self.distortion.append(0.0)
                    continue
                
                # Compute scale factor (how much the triangle was scaled in UV space)
                scale_factor = np.sqrt(area_uv / area_3d)
                
                # Conformal distortion: how much the scale deviates from ideal
                # Use log scale for better visualization
                distortion_val = abs(np.log(scale_factor))
                
                self.distortion.append(distortion_val)
                
            except Exception as e:
                # If anything fails, use neutral distortion
                self.distortion.append(0.0)
        
        # Auto-adjust distortion range for better visualization
        if self.distortion:
            distortions = np.array(self.distortion)
            if np.max(distortions) > 0:
                # Use 95th percentile to avoid outliers dominating the color scale
                q95 = np.percentile(distortions, 95)
                if q95 > 0:
                    self.distortion_range = (0.0, q95)
                else:
                    self.distortion_range = (0.0, np.max(distortions))
            else:
                self.distortion_range = (0.0, 1.0)
    
    def get_heatmap_color(self, distortion_value):
        """Convert distortion value to standard blue-to-red heatmap color"""
        # Normalize to current range
        min_val, max_val = self.distortion_range
        if max_val > min_val:
            normalized = (distortion_value - min_val) / (max_val - min_val)
        else:
            normalized = 0.0
        
        # Clamp to [0, 1]
        normalized = max(0.0, min(1.0, normalized))
        
        # Standard blue-to-red heatmap
        # Blue (0.0) -> Cyan -> Green -> Yellow -> Red (1.0)
        if normalized < 0.25:
            # Blue to Cyan
            t = normalized / 0.25
            r = 0.0
            g = t
            b = 1.0
        elif normalized < 0.5:
            # Cyan to Green
            t = (normalized - 0.25) / 0.25
            r = 0.0
            g = 1.0
            b = 1.0 - t
        elif normalized < 0.75:
            # Green to Yellow
            t = (normalized - 0.5) / 0.25
            r = t
            g = 1.0
            b = 0.0
        else:
            # Yellow to Red
            t = (normalized - 0.75) / 0.25
            r = 1.0
            g = 1.0 - t
            b = 0.0
            
        return r, g, b
        
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        
    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        padding = 0.1
        aspect = width / float(height) if height != 0 else 1.0
        
        if aspect > 1:
            glOrtho(-padding, 1 + padding, (-padding)/aspect, (1 + padding)/aspect, -1, 1)
        else:
            glOrtho(-padding * aspect, (1 + padding) * aspect, -padding, 1 + padding, -1, 1)
            
        glMatrixMode(GL_MODELVIEW)
        
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        if self.uv_vertices is not None and self.uv_faces is not None:
            self.draw_uv_layout()
            self.draw_uv_boundary()
                
    def draw_uv_layout(self):
        if self.uv_vertices is None or self.uv_faces is None:
            return
            
        if self.display_mode == "wireframe":
            # Wireframe only
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glColor3f(0.2, 0.2, 0.8)
            glLineWidth(1.5)
            
            glBegin(GL_TRIANGLES)
            for i, face in enumerate(self.uv_faces):
                if len(face) == 3:
                    for vertex_idx in face:
                        if vertex_idx < len(self.uv_vertices):
                            uv = self.uv_vertices[vertex_idx]
                            glVertex3f(uv[0], uv[1], 0)
            glEnd()
            
        elif self.display_mode == "heatmap" and self.distortion is not None:
            # Heatmap with optional wireframe overlay
            
            # PASS 1: Draw filled triangles with heatmap colors
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glBegin(GL_TRIANGLES)
            for i, face in enumerate(self.uv_faces):
                if len(face) == 3 and i < len(self.distortion):
                    # Heatmap color based on ACTUAL distortion
                    r, g, b = self.get_heatmap_color(self.distortion[i])
                    glColor3f(r, g, b)
                    
                    for vertex_idx in face:
                        if vertex_idx < len(self.uv_vertices):
                            uv = self.uv_vertices[vertex_idx]
                            glVertex3f(uv[0], uv[1], 0)
            glEnd()
            
            # PASS 2: Draw wireframe on top if enabled
            if self.show_wireframe_overlay:
                glDisable(GL_DEPTH_TEST)  # Make wireframe always visible
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glColor3f(0.0, 0.0, 0.0)  # Black wireframe for maximum contrast
                glLineWidth(1.2)  # Slightly thicker lines for visibility
                
                glBegin(GL_TRIANGLES)
                for i, face in enumerate(self.uv_faces):
                    if len(face) == 3:
                        for vertex_idx in face:
                            if vertex_idx < len(self.uv_vertices):
                                uv = self.uv_vertices[vertex_idx]
                                glVertex3f(uv[0], uv[1], 0)
                glEnd()
                
                glEnable(GL_DEPTH_TEST)  # Re-enable depth test
        
        # Reset to defaults
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glLineWidth(1.0)
        
    def draw_uv_boundary(self):
        glColor3f(0.7, 0.7, 0.7)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glVertex3f(0, 0, 0)
        glVertex3f(1, 0, 0)
        glVertex3f(1, 1, 0)
        glVertex3f(0, 1, 0)
        glEnd()
        
        # Grid
        glColor3f(0.9, 0.9, 0.9)
        glLineWidth(0.5)
        glBegin(GL_LINES)
        for i in range(1, 4):
            x = i * 0.25
            glVertex3f(x, 0, 0)
            glVertex3f(x, 1, 0)
        for i in range(1, 4):
            y = i * 0.25
            glVertex3f(0, y, 0)
            glVertex3f(1, y, 0)
        glEnd()
        
        glLineWidth(1.0)
        
    def set_display_mode(self, mode):
        """Set display mode: 'wireframe' or 'heatmap'"""
        self.display_mode = mode
        self.update()
        
    def set_wireframe_overlay(self, show_wireframe):
        """Set whether to show wireframe overlay in heatmap mode"""
        self.show_wireframe_overlay = show_wireframe
        self.update()

class ComparisonViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vertices_3d = None
        self.faces_3d = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        
        # Splitter for side-by-side view
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - 3D mesh
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(2, 2, 2, 2)
        left_layout.setSpacing(2)
        
        left_label = QLabel("3D Mesh • Drag to rotate • Wheel to zoom")
        left_label.setMaximumHeight(15)
        left_label.setAlignment(Qt.AlignCenter)
        left_label.setStyleSheet("color: gray; font-size: 9px;")
        left_layout.addWidget(left_label)
        
        self.mesh_viewer = MeshViewer3D()
        left_layout.addWidget(self.mesh_viewer)
        splitter.addWidget(left_widget)
        
        # Right side - UV layout
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(2, 2, 2, 2)
        right_layout.setSpacing(2)
        
        right_label = QLabel("UV Layout • Blue=low distortion, Red=high distortion")
        right_label.setMaximumHeight(15)
        right_label.setAlignment(Qt.AlignCenter)
        right_label.setStyleSheet("color: gray; font-size: 9px;")
        right_layout.addWidget(right_label)
        
        self.uv_viewer = UVLayoutViewer()
        right_layout.addWidget(self.uv_viewer)
        splitter.addWidget(right_widget)
        
        splitter.setSizes([500, 500])
        layout.addWidget(splitter, 1)
        
        # Controls
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(2, 2, 2, 2)
        
        # 3D view controls
        controls_layout.addWidget(QLabel("3D:"))
        self.wireframe_3d_checkbox = QCheckBox("Wireframe Overlay")
        self.wireframe_3d_checkbox.stateChanged.connect(self.on_3d_wireframe_changed)
        controls_layout.addWidget(self.wireframe_3d_checkbox)
        
        controls_layout.addSpacing(20)
        
        # UV view controls
        controls_layout.addWidget(QLabel("UV:"))
        
        self.uv_mode_combo = QComboBox()
        self.uv_mode_combo.addItems(["Wireframe", "Conformal Distortion"])
        self.uv_mode_combo.currentTextChanged.connect(self.on_uv_mode_changed)
        controls_layout.addWidget(self.uv_mode_combo)
        
        # UV wireframe overlay checkbox (only visible in heatmap mode)
        self.wireframe_uv_checkbox = QCheckBox("Show Wireframe")
        self.wireframe_uv_checkbox.setChecked(True)  # Default to on
        self.wireframe_uv_checkbox.stateChanged.connect(self.on_uv_wireframe_changed)
        self.wireframe_uv_checkbox.setVisible(False)  # Hidden initially
        controls_layout.addWidget(self.wireframe_uv_checkbox)
        
        controls_layout.addStretch()
        
        # Reset view button
        self.reset_view_btn = QPushButton("Reset Views")
        self.reset_view_btn.clicked.connect(self.reset_views)
        self.reset_view_btn.setMaximumWidth(80)
        controls_layout.addWidget(self.reset_view_btn)
        
        layout.addLayout(controls_layout)
        
    def set_mesh_data(self, vertices, faces, uv_vertices, uv_faces):
        """Set mesh data for both viewers"""
        self.vertices_3d = vertices
        self.faces_3d = faces
        self.mesh_viewer.set_mesh(vertices, faces)
        if uv_vertices is not None and uv_faces is not None:
            # Pass 3D data to compute distortion
            self.uv_viewer.set_uv_layout(uv_vertices, uv_faces, vertices, faces)
        
    def on_3d_wireframe_changed(self, state):
        wireframe = (state == Qt.Checked)
        self.mesh_viewer.set_wireframe_overlay(wireframe)
        
    def on_uv_mode_changed(self, mode_text):
        """Handle UV display mode change"""
        if mode_text == "Wireframe":
            self.uv_viewer.set_display_mode("wireframe")
            self.wireframe_uv_checkbox.setVisible(False)  # Hide wireframe option
        elif mode_text == "Conformal Distortion":
            self.uv_viewer.set_display_mode("heatmap")
            self.wireframe_uv_checkbox.setVisible(True)  # Show wireframe option
        
    def on_uv_wireframe_changed(self, state):
        """Handle UV wireframe overlay toggle"""
        show_wireframe = (state == Qt.Checked)
        self.uv_viewer.set_wireframe_overlay(show_wireframe)
        
    def reset_views(self):
        if hasattr(self.mesh_viewer, 'vertices') and self.mesh_viewer.vertices is not None:
            self.mesh_viewer.set_mesh(self.mesh_viewer.vertices, self.mesh_viewer.faces)
        
        if hasattr(self.uv_viewer, 'uv_vertices') and self.uv_viewer.uv_vertices is not None:
            self.uv_viewer.set_uv_layout(self.uv_viewer.uv_vertices, self.uv_viewer.uv_faces, 
                                       self.vertices_3d, self.faces_3d)
