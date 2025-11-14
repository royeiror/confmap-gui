import numpy as np
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QMouseEvent
from OpenGL.GL import *
from OpenGL.GLU import *

class MeshViewer(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vertices = None
        self.faces = None
        self.uv_vertices = None
        self.uv_faces = None
        
        # Camera controls
        self.rotation_x = 0
        self.rotation_y = 0
        self.zoom = -5
        self.last_pos = QPoint()
        
        # Display modes
        self.show_3d = True  # True for 3D, False for UV
        self.wireframe = False
        
    def set_mesh(self, vertices, faces, uv_vertices=None, uv_faces=None):
        """Set mesh data for display"""
        self.vertices = vertices
        self.faces = faces
        self.uv_vertices = uv_vertices
        self.uv_faces = uv_faces
        self.update()
        
    def initializeGL(self):
        """Initialize OpenGL"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set up lighting
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
        
        glClearColor(0.1, 0.1, 0.1, 1.0)
        
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
        
        if self.vertices is not None and self.faces is not None:
            if self.show_3d:
                self.draw_3d_mesh()
            else:
                self.draw_uv_layout()
                
    def draw_3d_mesh(self):
        """Draw the 3D mesh"""
        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDisable(GL_LIGHTING)
            glColor3f(0.8, 0.8, 0.8)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glEnable(GL_LIGHTING)
            glColor3f(0.7, 0.7, 0.9)
            
        glBegin(GL_TRIANGLES)
        for face in self.faces:
            if len(face) == 3:  # Triangle
                for vertex_idx in face:
                    if vertex_idx < len(self.vertices):
                        glVertex3fv(self.vertices[vertex_idx])
        glEnd()
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_LIGHTING)
        
    def draw_uv_layout(self):
        """Draw the UV layout in 2D"""
        if self.uv_vertices is None or self.uv_faces is None:
            return
            
        glDisable(GL_LIGHTING)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        
        # Scale UV coordinates to fit in view
        if len(self.uv_vertices) > 0:
            uv_array = np.array(self.uv_vertices)
            center = np.mean(uv_array, axis=0)
            max_extent = np.max(np.abs(uv_array - center)) * 1.2
            if max_extent > 0:
                scale = 2.0 / max_extent
            else:
                scale = 1.0
        else:
            center = [0, 0]
            scale = 1.0
            
        glColor3f(0.0, 0.8, 0.0)
        glBegin(GL_TRIANGLES)
        for face in self.uv_faces:
            if len(face) == 3:  # Triangle
                for uv_idx in face:
                    if uv_idx < len(self.uv_vertices):
                        uv = self.uv_vertices[uv_idx]
                        # Center and scale UV coordinates
                        x = (uv[0] - center[0]) * scale
                        y = (uv[1] - center[1]) * scale
                        glVertex3f(x, y, 0)
        glEnd()
        
        # Draw UV boundary
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_LINE_LOOP)
        glVertex3f(-1, -1, 0)
        glVertex3f(1, -1, 0)
        glVertex3f(1, 1, 0)
        glVertex3f(-1, 1, 0)
        glEnd()
        
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
        self.zoom += event.angleDelta().y() * 0.01
        self.zoom = max(-10, min(-1, self.zoom))  # Limit zoom
        self.update()
        
    def set_display_mode(self, show_3d):
        """Set display mode: True for 3D, False for UV"""
        self.show_3d = show_3d
        self.update()
        
    def set_wireframe(self, wireframe):
        """Set wireframe mode"""
        self.wireframe = wireframe
        self.update()
