import numpy as np
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QSplitter, QTextEdit)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent
from OpenGL.GL import *
from OpenGL.GLU import *

class SimpleUVViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UV Test - Direct Drawing")
        self.setGeometry(100, 100, 800, 600)
        
        layout = QVBoxLayout(self)
        
        # Create test button
        self.test_btn = QPushButton("Test UV Display")
        self.test_btn.clicked.connect(self.test_uv_display)
        layout.addWidget(self.test_btn)
        
        # Create OpenGL widget
        self.gl_widget = UVTestWidget()
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

class UVTestWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 400)
        
    def paintEvent(self, event):
        # Use QPainter to draw directly - bypass OpenGL completely
        from PyQt5.QtGui import QPainter, QColor, QPen
        from PyQt5.QtCore import QPointF
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), QColor(255, 255, 255))  # White
        
        # Draw UV boundary
        painter.setPen(QPen(QColor(0, 0, 255), 2))  # Blue
        boundary_rect = self.get_uv_rect()
        painter.drawRect(boundary_rect)
        
        # Draw test triangles
        self.draw_test_triangles(painter, boundary_rect)
        
        painter.end()
    
    def get_uv_rect(self):
        # Return a rectangle that represents the UV [0,1] space
        margin = 20
        size = min(self.width(), self.height()) - 2 * margin
        return self.rect().adjusted(margin, margin, -margin, -margin)
    
    def draw_test_triangles(self, painter, uv_rect):
        # Draw some test triangles that should definitely be visible
        
        # Convert UV coordinates to widget coordinates
        def uv_to_widget(uv_x, uv_y):
            x = uv_rect.left() + uv_x * uv_rect.width()
            y = uv_rect.top() + (1 - uv_y) * uv_rect.height()  # Flip Y
            return QPointF(x, y)
        
        # Triangle 1 - Large red triangle in center
        painter.setPen(QPen(QColor(255, 0, 0), 3))  # Red outline
        painter.setBrush(QColor(255, 200, 200))  # Light red fill
        points1 = [
            uv_to_widget(0.3, 0.3),
            uv_to_widget(0.7, 0.3), 
            uv_to_widget(0.5, 0.7)
        ]
        painter.drawPolygon(points1)
        
        # Triangle 2 - Green triangle in top-left
        painter.setPen(QPen(QColor(0, 255, 0), 3))  # Green outline
        painter.setBrush(QColor(200, 255, 200))  # Light green fill
        points2 = [
            uv_to_widget(0.1, 0.1),
            uv_to_widget(0.3, 0.1),
            uv_to_widget(0.1, 0.3)
        ]
        painter.drawPolygon(points2)
        
        # Triangle 3 - Blue triangle in bottom-right
        painter.setPen(QPen(QColor(0, 0, 255), 3))  # Blue outline
        painter.setBrush(QColor(200, 200, 255))  # Light blue fill
        points3 = [
            uv_to_widget(0.7, 0.7),
            uv_to_widget(0.9, 0.7),
            uv_to_widget(0.7, 0.9)
        ]
        painter.drawPolygon(points3)
    
    def test_draw_triangles(self):
        self.update()  # Force repaint

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimpleUVViewer()
    window.show()
    sys.exit(app.exec_())
