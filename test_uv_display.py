# test_uv_display.py
import numpy as np
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QSplitter, QTextEdit, QOpenGLWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent
from OpenGL.GL import *
from OpenGL.GLU import *

class SimpleOpenGLUVViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UV Test - Simple OpenGL")
        self.setGeometry(100, 100, 800, 600)
        
        layout = QVBoxLayout(self)
        
        # Create test button
        self.test_btn = QPushButton("Test UV Display")
        self.test_btn.clicked.connect(self.test_uv_display)
        layout.addWidget(self.test_btn)
        
        # Create OpenGL widget
        self.gl_widget = SimpleUVOpenGLWidget()
        layout.addWidget(self.gl_widget)
        
        # Log
        self.log = QTextEdit()
        self.log.setMaximumHeight(100)
        layout.addWidget(self.log)
        
        self.log_message("Application started. Click 'Test UV Display'")
    
    def log_message(self, message):
        self.log.append(message)
        print(message)
    
    def test_uv_display(self):
        self.log_message("Testing UV display...")
        self.gl_widget.test_draw_triangles()

class SimpleUVOpenGLWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.test_triangles = None
        
    def initializeGL(self):
        print("OpenGL initialized")
        glClearColor(1.0, 1.0, 1.0, 1.0)  # White background
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
    def resizeGL(self, width, height):
        print(f"Resize: {width}x{height}")
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, 1, 0, 1, -1, 1)  # Simple 2D projection
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
    def paintGL(self):
        print("Painting...")
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        
        # Draw boundary
        self.draw_boundary()
        
        # Draw test triangles if available
        if self.test_triangles is not None:
            self.draw_triangles()
        else:
            self.draw_placeholder()
            
        print("Painting complete")
    
    def draw_boundary(self):
        glColor3f(0.0, 0.0, 1.0)  # Blue
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(0.0, 0.0)
        glVertex2f(1.0, 0.0)
        glVertex2f(1.0, 1.0)
        glVertex2f(0.0, 1.0)
        glEnd()
        glLineWidth(1.0)
        
        # Draw grid
        glColor3f(0.8, 0.8, 0.8)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for i in range(1, 4):
            x = i * 0.25
            glVertex2f(x, 0.0)
            glVertex2f(x, 1.0)
            glVertex2f(0.0, x)
            glVertex2f(1.0, x)
        glEnd()
    
    def draw_triangles(self):
        print("Drawing triangles...")
        
        # Draw filled triangles
        glColor3f(1.0, 0.0, 0.0)  # Red
        glBegin(GL_TRIANGLES)
        for triangle in self.test_triangles:
            for vertex in triangle:
                glVertex2f(vertex[0], vertex[1])
        glEnd()
        
        # Draw wireframe
        glColor3f(0.0, 0.0, 0.0)  # Black
        glLineWidth(2.0)
        glBegin(GL_LINES)
        for triangle in self.test_triangles:
            for i in range(3):
                v1 = triangle[i]
                v2 = triangle[(i + 1) % 3]
                glVertex2f(v1[0], v1[1])
                glVertex2f(v2[0], v2[1])
        glEnd()
        glLineWidth(1.0)
    
    def draw_placeholder(self):
        glColor3f(0.5, 0.5, 0.5)
        glLineWidth(3.0)
        glBegin(GL_LINES)
        glVertex2f(0.1, 0.1)
        glVertex2f(0.9, 0.9)
        glVertex2f(0.9, 0.1)
        glVertex2f(0.1, 0.9)
        glEnd()
        glLineWidth(1.0)
    
    def test_draw_triangles(self):
        print("Setting up test triangles...")
        # Create some test triangles that should definitely be visible
        self.test_triangles = [
            # Large triangle in center
            [(0.3, 0.3), (0.7, 0.3), (0.5, 0.7)],
            # Small triangle in corner
            [(0.1, 0.1), (0.3, 0.1), (0.1, 0.3)],
            # Another triangle
            [(0.7, 0.7), (0.9, 0.7), (0.7, 0.9)]
        ]
        print(f"Test triangles: {self.test_triangles}")
        self.update()  # Force repaint

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimpleOpenGLUVViewer()
    window.show()
    sys.exit(app.exec_())
