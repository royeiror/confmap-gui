import numpy as np
from PyQt5.QtWidgets import (QOpenGLWidget, QVBoxLayout, QHBoxLayout, QWidget, 
                             QLabel, QSplitter, QCheckBox, QPushButton)
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
        self.far_plane = 1000.0  # Much larger far plane
        
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
            
            # Auto-adjust zoom to fit mesh with some padding
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
        
        # Set up secondary lighting for better depth perception
        glLightfv(GL_LIGHT1, GL_POSITION, [-2, -1, -1, 1])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.3, 0.3, 0.3, 1])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.2, 0.2, 0.2, 1])
        
        # Set material properties
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.5, 0.5, 0.5, 1])
        glMaterialf(GL_FRONT, GL_SHININESS, 50.0)
        
        glClearColor(0.1, 0.1, 0.1, 1.0)
        
    def resizeGL(self, width, height):
        """Handle window resize"""
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = width / float(height) if height != 0 else 1.0
        # Use much larger far plane for unlimited zoom out
        gluPerspective(45, aspect, self.near_plane, self.far_plane)
        glMatrixMode(GL_MODELVIEW)
        
    def paintGL(self):
        """Render the scene"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Camera position - no limits on zoom
        glTranslatef(0, 0, self.zoom)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        
        # Center the mesh
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
                # Calculate face normal for better lighting
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
        """Handle mouse wheel for zoom - NO LIMITS"""
        zoom_speed = max(0.1, abs(self.zoom) * 0.05)
        self.zoom += event.angleDelta().y() * 0.001 * zoom_speed
        self.update()
        
    def set_wireframe_overlay(self, wireframe):
        """Set wireframe overlay mode"""
        self.wireframe_overlay = wireframe
        self.update()
