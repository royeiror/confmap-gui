import numpy as np
import xml.etree.ElementTree as ET
from PyQt5.QtWidgets import (QOpenGLWidget, QVBoxLayout, QHBoxLayout, QWidget, 
                             QLabel, QSplitter, QCheckBox, QPushButton, QComboBox,
                             QFileDialog, QMessageBox, QSlider, QSpinBox)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QMouseEvent
from OpenGL.GL import *
from OpenGL.GLU import *
import os
import tempfile

class MeshViewer3D(QOpenGLWidget):
    # ... (keep the existing MeshViewer3D class exactly as before) ...

class UVLayoutViewer(QOpenGLWidget):
    # ... (keep the existing UVLayoutViewer class exactly as before) ...

class ComparisonViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vertices_3d = None
        self.faces_3d = None
        self.uv_vertices = None
        self.uv_faces = None
        self.svg_scale = 100.0  # Pixels per unit for SVG export
        self.seam_allowance = 0.0  # Additional border around pieces
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
        
        right_label = QLabel("UV Layout • Full spectrum: Blue (low) to Red (high distortion)")
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
        
        # Export button
        self.export_btn = QPushButton("Export Fabric Pattern")
        self.export_btn.clicked.connect(self.export_fabric_pattern)
        self.export_btn.setMaximumWidth(120)
        controls_layout.addWidget(self.export_btn)
        
        # Reset view button
        self.reset_view_btn = QPushButton("Reset Views")
        self.reset_view_btn.clicked.connect(self.reset_views)
        self.reset_view_btn.setMaximumWidth(80)
        controls_layout.addWidget(self.reset_view_btn)
        
        layout.addLayout(controls_layout)
        
        # SVG Export settings
        export_layout = QHBoxLayout()
        export_layout.setContentsMargins(2, 2, 2, 2)
        
        export_layout.addWidget(QLabel("SVG Scale (px/cm):"))
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setMinimum(50)
        self.scale_slider.setMaximum(200)
        self.scale_slider.setValue(100)
        self.scale_slider.valueChanged.connect(self.on_scale_changed)
        export_layout.addWidget(self.scale_slider)
        
        self.scale_label = QLabel("100")
        self.scale_label.setMaximumWidth(30)
        export_layout.addWidget(self.scale_label)
        
        export_layout.addSpacing(20)
        
        export_layout.addWidget(QLabel("Seam Allowance (mm):"))
        self.seam_spinbox = QSpinBox()
        self.seam_spinbox.setMinimum(0)
        self.seam_spinbox.setMaximum(20)
        self.seam_spinbox.setValue(0)
        self.seam_spinbox.valueChanged.connect(self.on_seam_allowance_changed)
        export_layout.addWidget(self.seam_spinbox)
        
        export_layout.addStretch()
        
        layout.addLayout(export_layout)
        
    def set_mesh_data(self, vertices, faces, uv_vertices, uv_faces):
        """Set mesh data for both viewers"""
        self.vertices_3d = vertices
        self.faces_3d = faces
        self.uv_vertices = uv_vertices
        self.uv_faces = uv_faces
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
        
    def on_scale_changed(self, value):
        self.svg_scale = value
        self.scale_label.setText(str(value))
        
    def on_seam_allowance_changed(self, value):
        self.seam_allowance = value / 10.0  # Convert mm to cm
        
    def reset_views(self):
        if hasattr(self.mesh_viewer, 'vertices') and self.mesh_viewer.vertices is not None:
            self.mesh_viewer.set_mesh(self.mesh_viewer.vertices, self.mesh_viewer.faces)
        
        if hasattr(self.uv_viewer, 'uv_vertices') and self.uv_viewer.uv_vertices is not None:
            self.uv_viewer.set_uv_layout(self.uv_viewer.uv_vertices, self.uv_viewer.uv_faces, 
                                       self.vertices_3d, self.faces_3d)

    def export_fabric_pattern(self):
        """Export UV layout as SVG for fabric forming"""
        if self.uv_vertices is None or self.uv_faces is None:
            QMessageBox.warning(self, "Export Error", "No UV layout data available")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Fabric Pattern", "", "SVG Files (*.svg)")
            
        if not filename:
            return
            
        try:
            self.create_fabric_svg(filename)
            QMessageBox.information(self, "Export Successful", 
                                  f"Fabric pattern saved to:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export SVG: {str(e)}")

    def create_fabric_svg(self, filename):
        """Create SVG file optimized for fabric forming"""
        # Calculate bounding box of UV layout
        if len(self.uv_vertices) == 0:
            raise ValueError("No UV vertices available")
            
        u_coords = self.uv_vertices[:, 0]
        v_coords = self.uv_vertices[:, 1]
        
        min_u, max_u = np.min(u_coords), np.max(u_coords)
        min_v, max_v = np.min(v_coords), np.max(v_coords)
        
        width = (max_u - min_u) * self.svg_scale
        height = (max_v - min_v) * self.svg_scale
        
        # Add margin for seam allowance and registration marks
        margin = self.seam_allowance * self.svg_scale + 50  # 50px for registration marks
        total_width = width + 2 * margin
        total_height = height + 2 * margin
        
        # Create SVG root
        svg = ET.Element('svg', {
            'xmlns': 'http://www.w3.org/2000/svg',
            'width': f'{total_width}px',
            'height': f'{total_height}px',
            'viewBox': f'0 0 {total_width} {total_height}'
        })
        
        # Add background
        background = ET.SubElement(svg, 'rect', {
            'x': '0', 'y': '0', 
            'width': str(total_width), 
            'height': str(total_height),
            'fill': 'white'
        })
        
        # Add title and description
        title = ET.SubElement(svg, 'title')
        title.text = 'Fabric Forming Pattern'
        
        desc = ET.SubElement(svg, 'desc')
        desc.text = f'UV conformal map for fabric forming. Scale: {self.svg_scale}px/cm, Seam allowance: {self.seam_allowance}cm'
        
        # Transform group for the actual pattern
        pattern_group = ET.SubElement(svg, 'g', {
            'transform': f'translate({margin} {margin}) scale({self.svg_scale} -{self.svg_scale}) translate(0 -{max_v - min_v})'
        })
        
        # Draw triangles with optional seam allowance
        for face in self.uv_faces:
            if len(face) == 3:
                points = []
                for vertex_idx in face:
                    if vertex_idx < len(self.uv_vertices):
                        uv = self.uv_vertices[vertex_idx]
                        # Apply seam allowance by offsetting points outward from center
                        if self.seam_allowance > 0:
                            # Calculate face center
                            face_uvs = [self.uv_vertices[face[i]] for i in range(3)]
                            center = np.mean(face_uvs, axis=0)
                            # Offset point away from center
                            direction = uv - center
                            dir_length = np.linalg.norm(direction)
                            if dir_length > 1e-10:
                                offset_uv = uv + (direction / dir_length) * self.seam_allowance
                                points.append(f"{offset_uv[0] - min_u},{offset_uv[1] - min_v}")
                            else:
                                points.append(f"{uv[0] - min_u},{uv[1] - min_v}")
                        else:
                            points.append(f"{uv[0] - min_u},{uv[1] - min_v}")
                
                if len(points) == 3:
                    # Create triangle
                    triangle = ET.SubElement(pattern_group, 'polygon', {
                        'points': ' '.join(points),
                        'fill': 'none',
                        'stroke': 'black',
                        'stroke-width': str(0.5 / self.svg_scale),  # Scale line width
                        'stroke-linejoin': 'round'
                    })
        
        # Add registration marks
        self.add_registration_marks(svg, total_width, total_height, margin)
        
        # Add scale indicator
        self.add_scale_indicator(svg, total_width, total_height)
        
        # Add cutting instructions
        self.add_cutting_instructions(svg, total_width, total_height)
        
        # Write SVG file
        tree = ET.ElementTree(svg)
        tree.write(filename, encoding='utf-8', xml_declaration=True)

    def add_registration_marks(self, svg, width, height, margin):
        """Add registration marks for alignment"""
        marks = [
            (margin/2, margin/2),  # Top-left
            (width - margin/2, margin/2),  # Top-right
            (margin/2, height - margin/2),  # Bottom-left
            (width - margin/2, height - margin/2),  # Bottom-right
            (width/2, margin/2),  # Top-center
            (width/2, height - margin/2),  # Bottom-center
        ]
        
        for x, y in marks:
            # Crosshair
            ET.SubElement(svg, 'line', {
                'x1': str(x - 10), 'y1': str(y),
                'x2': str(x + 10), 'y2': str(y),
                'stroke': 'red', 'stroke-width': '1'
            })
            ET.SubElement(svg, 'line', {
                'x1': str(x), 'y1': str(y - 10),
                'x2': str(x), 'y2': str(y + 10),
                'stroke': 'red', 'stroke-width': '1'
            })
            # Circle
            ET.SubElement(svg, 'circle', {
                'cx': str(x), 'cy': str(y), 'r': '8',
                'fill': 'none', 'stroke': 'red', 'stroke-width': '1'
            })

    def add_scale_indicator(self, svg, width, height):
        """Add scale indicator bar"""
        bar_length = 5.0 * self.svg_scale  # 5cm scale bar
        bar_x = 20
        bar_y = height - 40
        
        # Scale bar
        ET.SubElement(svg, 'line', {
            'x1': str(bar_x), 'y1': str(bar_y),
            'x2': str(bar_x + bar_length), 'y2': str(bar_y),
            'stroke': 'black', 'stroke-width': '3'
        })
        
        # Labels
        ET.SubElement(svg, 'text', {
            'x': str(bar_x), 'y': str(bar_y - 10),
            'font-family': 'Arial', 'font-size': '12', 'fill': 'black'
        }).text = '0'
        
        ET.SubElement(svg, 'text', {
            'x': str(bar_x + bar_length), 'y': str(bar_y - 10),
            'font-family': 'Arial', 'font-size': '12', 'fill': 'black',
            'text-anchor': 'end'
        }).text = '5 cm'
        
        # Title
        ET.SubElement(svg, 'text', {
            'x': str(bar_x), 'y': str(bar_y - 25),
            'font-family': 'Arial', 'font-size': '14', 'fill': 'black',
            'font-weight': 'bold'
        }).text = 'Scale Indicator'

    def add_cutting_instructions(self, svg, width, height):
        """Add cutting and assembly instructions"""
        instructions = [
            "FABRIC FORMING PATTERN INSTRUCTIONS:",
            "1. Print this pattern at 100% scale",
            "2. Cut along all black lines",
            "3. Place on pre-stretched fabric",
            "4. Adhere pattern to fabric",
            "5. Release fabric tension to form 3D shape"
        ]
        
        text_x = 20
        text_y = 30
        
        for i, line in enumerate(instructions):
            ET.SubElement(svg, 'text', {
                'x': str(text_x), 'y': str(text_y + i * 18),
                'font-family': 'Arial', 'font-size': '12', 'fill': 'blue',
                'font-weight': 'bold' if i == 0 else 'normal'
            }).text = line

# Update the main window to include the comparison viewer
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UV Conformal Map Tool - Fabric Forming")
        self.setGeometry(100, 100, 1400, 800)
        
        layout = QVBoxLayout(self)
        
        # Add comparison viewer
        self.comparison_viewer = ComparisonViewer()
        layout.addWidget(self.comparison_viewer)
        
        # Example usage - you would replace this with your actual mesh loading
        # self.load_example_mesh()

    def load_example_mesh(self):
        """Load an example mesh for testing"""
        # This would be replaced with your actual mesh loading code
        # For now, create a simple cube
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ], dtype=np.float32)
        
        faces = np.array([
            [0, 1, 2], [2, 3, 0],  # front
            [1, 5, 6], [6, 2, 1],  # right
            [5, 4, 7], [7, 6, 5],  # back
            [4, 0, 3], [3, 7, 4],  # left
            [3, 2, 6], [6, 7, 3],  # top
            [4, 5, 1], [1, 0, 4]   # bottom
        ], dtype=np.int32)
        
        # Simple UV mapping (this would come from your conformal mapping)
        uv_vertices = np.array([
            [0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.4],
            [0.6, 0.2], [0.8, 0.2], [0.8, 0.4], [0.6, 0.4]
        ], dtype=np.float32)
        
        uv_faces = faces  # Same connectivity
        
        self.comparison_viewer.set_mesh_data(vertices, faces, uv_vertices, uv_faces)
