import numpy as np
import xml.etree.ElementTree as ET
from PyQt5.QtWidgets import (QOpenGLWidget, QVBoxLayout, QHBoxLayout, QWidget, 
                             QLabel, QSplitter, QCheckBox, QPushButton, QComboBox,
                             QFileDialog, QMessageBox, QSlider, QSpinBox, QApplication,
                             QProgressBar, QTextEdit, QTabWidget, QGroupBox)
from PyQt5.QtCore import Qt, QPoint, QThread, pyqtSignal, QRectF
from PyQt5.QtGui import QMouseEvent
from OpenGL.GL import *
from OpenGL.GLU import *
import os
import tempfile
import sys
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import time
import math

class SimpleObjLoader:
    """Simple OBJ file loader that doesn't require trimesh"""
    
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
                            # Handle format: vertex/texture/normal or just vertex
                            vertex_data = part.split('/')[0]
                            if vertex_data:
                                try:
                                    # OBJ indices are 1-based, convert to 0-based
                                    vertex_idx = int(vertex_data) - 1
                                    if vertex_idx >= 0:
                                        face_vertices.append(vertex_idx)
                                except ValueError:
                                    continue
                        
                        if len(face_vertices) >= 3:
                            # Convert to triangles if it's a quad or more
                            for i in range(1, len(face_vertices) - 1):
                                faces.append([face_vertices[0], face_vertices[i], face_vertices[i + 1]])
            
            return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)
            
        except Exception as e:
            raise ValueError(f"Error parsing OBJ file: {str(e)}")

class TrianglePacker:
    """Packs triangles efficiently for fabric cutting patterns"""
    
    @staticmethod
    def pack_triangles(triangles_3d, uv_vertices, uv_faces, padding=0.02):
        """Pack triangles into a rectangular area for efficient fabric usage"""
        # Extract individual triangles with their original 3D data
        triangle_islands = []
        
        for i, face in enumerate(uv_faces):
            if len(face) == 3:
                # Get UV coordinates for this triangle
                uv_triangle = [uv_vertices[face[0]], uv_vertices[face[1]], uv_vertices[face[2]]]
                
                # Get corresponding 3D triangle for distortion computation
                if i < len(triangles_3d):
                    triangle_3d = triangles_3d[i]
                else:
                    triangle_3d = None
                
                triangle_islands.append({
                    'uv_points': np.array(uv_triangle),
                    'face_index': i,
                    '3d_data': triangle_3d
                })
        
        # Sort triangles by area (largest first for better packing)
        triangle_islands.sort(key=lambda x: TrianglePacker.triangle_area(x['uv_points']), reverse=True)
        
        # Pack triangles
        packed_triangles = []
        current_x = 0.0
        current_y = 0.0
        row_height = 0.0
        max_width = 0.0
        
        for triangle in triangle_islands:
            points = triangle['uv_points']
            
            # Calculate triangle bounding box
            min_x, min_y = np.min(points, axis=0)
            max_x, max_y = np.max(points, axis=0)
            width = max_x - min_x
            height = max_y - min_y
            
            # Check if triangle fits in current row
            if current_x + width + padding > 1.0:  # Start new row
                current_x = 0.0
                current_y += row_height + padding
                row_height = 0.0
            
            # Position triangle
            offset_x = current_x - min_x
            offset_y = current_y - min_y
            
            # Update row height
            row_height = max(row_height, height)
            max_width = max(max_width, current_x + width)
            
            # Store packed triangle
            packed_points = points + np.array([offset_x, offset_y])
            packed_triangles.append({
                'original_face_index': triangle['face_index'],
                'packed_points': packed_points,
                '3d_data': triangle['3d_data']
            })
            
            current_x += width + padding
        
        # Normalize all points to fit in [0,1] range
        scale_factor = max(max_width, current_y + row_height)
        if scale_factor > 0:
            for triangle in packed_triangles:
                triangle['packed_points'] /= scale_factor
        
        return packed_triangles

    @staticmethod
    def triangle_area(points):
        """Calculate area of a triangle"""
        if len(points) != 3:
            return 0.0
        a = points[0]
        b = points[1]
        c = points[2]
        return 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))

class ConformalMappingThread(QThread):
    """Thread for computing conformal mapping to avoid GUI freezing"""
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
        """Compute per-face conformal mapping with individual triangle parameterization"""
        self.progress_signal.emit(10)
        self.log_signal.emit("Computing individual triangle mappings...")
        
        n_faces = len(self.faces)
        triangles_3d = []
        uv_vertices = []
        uv_faces = []
        vertex_offset = 0
        
        for face_idx, face in enumerate(self.faces):
            if len(face) != 3:
                continue
                
            self.progress_signal.emit(10 + int(80 * face_idx / n_faces))
            
            # Get 3D triangle
            v0_3d, v1_3d, v2_3d = self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]
            triangles_3d.append((v0_3d, v1_3d, v2_3d))
            
            # Simple planar parameterization for each triangle
            # Use the triangle's plane coordinates
            edge1 = v1_3d - v0_3d
            edge2 = v2_3d - v0_3d
            
            # Create orthonormal basis in triangle's plane
            normal = np.cross(edge1, edge2)
            if np.linalg.norm(normal) < 1e-10:
                # Degenerate triangle, use default UVs
                uv_triangle = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
            else:
                normal = normal / np.linalg.norm(normal)
                
                # Choose basis vectors in the plane
                u_axis = edge1 / np.linalg.norm(edge1)
                v_axis = np.cross(normal, u_axis)
                v_axis = v_axis / np.linalg.norm(v_axis)
                
                # Project vertices to 2D
                uv0 = np.array([0, 0])
                uv1 = np.array([np.linalg.norm(edge1), 0])
                uv2 = np.array([np.dot(edge2, u_axis), np.dot(edge2, v_axis)])
                
                uv_triangle = np.array([uv0, uv1, uv2])
            
            # Add to UV arrays
            uv_vertices.extend(uv_triangle)
            uv_faces.append([vertex_offset, vertex_offset + 1, vertex_offset + 2])
            vertex_offset += 3
        
        self.progress_signal.emit(90)
        self.log_signal.emit("Packing triangles for fabric cutting...")
        
        # Pack triangles efficiently
        packed_data = TrianglePacker.pack_triangles(triangles_3d, np.array(uv_vertices), uv_faces)
        
        # Rebuild UV arrays from packed data
        packed_uv_vertices = []
        packed_uv_faces = []
        new_vertex_offset = 0
        
        for packed_triangle in packed_data:
            points = packed_triangle['packed_points']
            packed_uv_vertices.extend(points)
            packed_uv_faces.append([new_vertex_offset, new_vertex_offset + 1, new_vertex_offset + 2])
            new_vertex_offset += 3
        
        self.progress_signal.emit(100)
        return np.array(packed_uv_vertices, dtype=np.float32), packed_uv_faces, triangles_3d

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
        """Compute conformal distortion for each face using scale factors"""
        self.distortion = []
        
        for i, face_3d in enumerate(faces_3d):
            if len(face_3d) != 3:
                self.distortion.append(0.0)
                continue
                
            try:
                # Get 3D triangle vertices
                v0_3d, v1_3d, v2_3d = vertices_3d[face_3d[0]], vertices_3d[face_3d[1]], vertices_3d[face_3d[2]]
                
                # Get corresponding UV triangle vertices
                if i < len(uv_faces):
                    uv_face = uv_faces[i]
                else:
                    uv_face = face_3d
                    
                # Ensure we have valid UV indices
                if (uv_face[0] >= len(uv_vertices) or uv_face[1] >= len(uv_vertices) or 
                    uv_face[2] >= len(uv_vertices)):
                    self.distortion.append(0.0)
                    continue
                    
                uv0, uv1, uv2 = uv_vertices[uv_face[0]], uv_vertices[uv_face[1]], uv_vertices[uv_face[2]]
                
                # Compute edge vectors in 3D
                e1_3d = v1_3d - v0_3d
                e2_3d = v2_3d - v0_3d
                
                # Compute edge vectors in UV
                e1_uv = uv1 - uv0
                e2_uv = uv2 - uv0
                
                # Compute areas
                area_3d = 0.5 * np.linalg.norm(np.cross(e1_3d, e2_3d))
                area_uv = 0.5 * abs(e1_uv[0]*e2_uv[1] - e1_uv[1]*e2_uv[0])
                
                if area_3d < 1e-10 or area_uv < 1e-10:
                    self.distortion.append(0.0)
                    continue
                
                # Compute edge lengths in 3D and UV
                len_e1_3d = np.linalg.norm(e1_3d)
                len_e2_3d = np.linalg.norm(e2_3d)
                len_e1_uv = np.linalg.norm(e1_uv)
                len_e2_uv = np.linalg.norm(e2_uv)
                
                # Compute scale factors for both edges
                scale1 = len_e1_uv / len_e1_3d if len_e1_3d > 1e-10 else 1.0
                scale2 = len_e2_uv / len_e2_3d if len_e2_3d > 1e-10 else 1.0
                
                # Compute area scale factor
                area_scale = area_uv / area_3d
                
                # Conformal distortion: measure of how different the scales are
                # Perfect conformal mapping would have uniform scaling
                max_scale = max(scale1, scale2, area_scale)
                min_scale = min(scale1, scale2, area_scale)
                
                if min_scale > 1e-10:
                    scale_ratio = max_scale / min_scale
                    # Log scale to make it more manageable, offset by 1 to avoid log(1) = 0
                    distortion_val = np.log(scale_ratio)
                else:
                    distortion_val = 0.0
                
                self.distortion.append(distortion_val)
                
            except Exception as e:
                print(f"Error computing distortion for face {i}: {e}")
                self.distortion.append(0.0)
        
        # Auto-adjust distortion range to ensure good color variation
        if self.distortion:
            distortions = np.array(self.distortion)
            valid_distortions = distortions[np.isfinite(distortions)]
            
            if len(valid_distortions) > 0:
                # Use percentiles to remove outliers
                q10 = np.percentile(valid_distortions, 10)
                q90 = np.percentile(valid_distortions, 90)
                
                if q90 > q10:
                    # Add some padding
                    padding = (q90 - q10) * 0.2
                    self.distortion_range = (max(0, q10 - padding), q90 + padding)
                else:
                    # If all values are similar, use min/max with padding
                    min_val = np.min(valid_distortions)
                    max_val = np.max(valid_distortions)
                    if max_val > min_val:
                        padding = (max_val - min_val) * 0.2
                        self.distortion_range = (min_val - padding, max_val + padding)
                    else:
                        self.distortion_range = (0.0, 1.0)
            else:
                self.distortion_range = (0.0, 1.0)
            
            print(f"Distortion range: {self.distortion_range}")
            print(f"Distortion stats - Min: {np.min(valid_distortions) if len(valid_distortions) > 0 else 'N/A'}, "
                  f"Max: {np.max(valid_distortions) if len(valid_distortions) > 0 else 'N/A'}, "
                  f"Mean: {np.mean(valid_distortions) if len(valid_distortions) > 0 else 'N/A'}")
    
    def get_heatmap_color(self, distortion_value):
        """Convert distortion value to heatmap color with better distribution"""
        min_val, max_val = self.distortion_range
        
        # If range is too small, use a fixed range to ensure colors
        if max_val - min_val < 1e-10:
            normalized = 0.5
        else:
            normalized = (distortion_value - min_val) / (max_val - min_val)
        
        # Clamp to [0, 1]
        normalized = max(0.0, min(1.0, normalized))
        
        # Enhanced heatmap with more visible color transitions
        if normalized < 0.2:
            # Deep blue to blue
            t = normalized / 0.2
            r = 0.0
            g = 0.0
            b = 0.5 + 0.5 * t
        elif normalized < 0.4:
            # Blue to cyan
            t = (normalized - 0.2) / 0.2
            r = 0.0
            g = t
            b = 1.0
        elif normalized < 0.6:
            # Cyan to green
            t = (normalized - 0.4) / 0.2
            r = 0.0
            g = 1.0
            b = 1.0 - t
        elif normalized < 0.8:
            # Green to yellow
            t = (normalized - 0.6) / 0.2
            r = t
            g = 1.0
            b = 0.0
        else:
            # Yellow to red
            t = (normalized - 0.8) / 0.2
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
        
        # Instructions tab
        instructions_tab = QWidget()
        instructions_layout = QVBoxLayout(instructions_tab)
        
        instructions = """
        <h3>UV Conformal Map Tool - Instructions</h3>
        
        <h4>Step 1: Load 3D Model</h4>
        <ul>
        <li>Click 'Load 3D Model' to import your 3D mesh</li>
        <li>Supported format: OBJ files only</li>
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
        
        self.log("Application started. Load an OBJ file to begin.")
        
    def log(self, message):
        """Add message to log with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
    def load_model(self):
        """Load a 3D model file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open 3D Model", "", 
            "OBJ Files (*.obj);;All Files (*)"
        )
        
        if not filename:
            return
            
        try:
            self.log(f"Loading model: {os.path.basename(filename)}")
            self.file_label.setText(f"Loaded: {os.path.basename(filename)}")
            
            # Load mesh using simple OBJ loader
            self.vertices, self.faces = SimpleObjLoader.load_obj(filename)
            
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
    
    def on_mapping_finished(self, uv_vertices, uv_faces, triangles_3d):
        """Handle completion of UV mapping"""
        # Re-enable buttons
        self.load_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if uv_vertices is not None and uv_faces is not None:
            self.uv_vertices = uv_vertices
            self.uv_faces = uv_faces
            self.comparison_viewer.triangles_3d = triangles_3d
            
            # Update comparison viewer with both 3D and UV data
            self.comparison_viewer.set_mesh_data(
                self.vertices, self.faces, 
                self.uv_vertices, self.uv_faces
            )
            
            self.log("UV mapping completed. Triangles packed for fabric cutting.")
            self.log(f"Generated {len(uv_faces)} triangle islands for fabric forming.")
            
            # Switch to wireframe view to see the packed triangles
            self.comparison_viewer.uv_mode_combo.setCurrentText("Wireframe")
            
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
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
