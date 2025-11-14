"""
Minimal implementation of BFF (Boundary First Flattening) for OBJ file processing
This avoids the heavy dependencies of the original confmap library
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import re

def read_obj(file_path):
    """Read vertices and faces from OBJ file"""
    vertices = []
    faces = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                # Vertex: v x y z
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                # Face: f v1 v2 v3 ...
                parts = line.split()
                face_vertices = []
                for part in parts[1:]:
                    # Handle format: vertex/texture/normal or just vertex
                    vertex_index = part.split('/')[0]
                    if vertex_index:
                        face_vertices.append(int(vertex_index) - 1)  # OBJ is 1-indexed
                if len(face_vertices) >= 3:
                    faces.append(face_vertices)
    
    return np.array(vertices, dtype=np.float64), faces

def write_obj(file_path, vertices, faces, uv_vertices=None, uv_faces=None):
    """Write vertices, faces, and optional UV coordinates to OBJ file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write UV vertices if provided
        if uv_vertices is not None:
            for uv in uv_vertices:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f} 0.0\n")
        
        # Write faces
        if uv_vertices is not None and uv_faces is not None:
            # Write faces with texture coordinates
            for i, face in enumerate(faces):
                uv_face = uv_faces[i]
                face_line = "f"
                for j in range(len(face)):
                    face_line += f" {face[j] + 1}/{uv_face[j] + 1}"
                f.write(face_line + "\n")
        else:
            # Write faces without texture coordinates
            for face in faces:
                face_line = "f"
                for vertex_idx in face:
                    face_line += f" {vertex_idx + 1}"
                f.write(face_line + "\n")

class MinimalBFF:
    """Minimal implementation of Boundary First Flattening"""
    
    def __init__(self, vertices, faces):
        self.vertices = vertices.copy()
        self.faces = faces
        self.uv_vertices = None
        self.uv_faces = None
        
    def find_boundary_vertices(self):
        """Find boundary vertices of the mesh"""
        # Count edge occurrences
        edge_count = {}
        for face in self.faces:
            n = len(face)
            for i in range(n):
                edge = tuple(sorted([face[i], face[(i + 1) % n]]))
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # Boundary edges appear only once
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        
        if not boundary_edges:
            raise ValueError("No boundary found - mesh might be closed")
        
        # Reconstruct boundary loop
        boundary_vertices = []
        current_edge = boundary_edges[0]
        boundary_vertices.extend(current_edge)
        
        remaining_edges = boundary_edges[1:]
        
        while remaining_edges:
            found = False
            for i, edge in enumerate(remaining_edges):
                if edge[0] == boundary_vertices[-1]:
                    boundary_vertices.append(edge[1])
                    remaining_edges.pop(i)
                    found = True
                    break
                elif edge[1] == boundary_vertices[-1]:
                    boundary_vertices.append(edge[0])
                    remaining_edges.pop(i)
                    found = True
                    break
            
            if not found:
                break
        
        return boundary_vertices
    
    def compute_cotangent_weights(self):
        """Compute cotangent weights for Laplace-Beltrami operator"""
        n_vertices = len(self.vertices)
        I = []
        J = []
        V = []
        
        for face in self.faces:
            if len(face) == 3:  # Triangle faces
                i, j, k = face
                
                # Compute edge vectors
                vi, vj, vk = self.vertices[i], self.vertices[j], self.vertices[k]
                
                # Compute cotangent weights
                e_ij = vj - vi
                e_ik = vk - vi
                e_ji = vi - vj
                e_jk = vk - vj
                e_ki = vi - vk
                e_kj = vj - vk
                
                # Cotangent of angle at vertex k
                cot_k = np.dot(e_ij, -e_ki) / np.linalg.norm(np.cross(e_ij, e_ki))
                
                # Cotangent of angle at vertex j  
                cot_j = np.dot(e_ik, -e_jk) / np.linalg.norm(np.cross(e_ik, e_jk))
                
                # Cotangent of angle at vertex i
                cot_i = np.dot(e_jk, -e_ij) / np.linalg.norm(np.cross(e_jk, e_ij))
                
                # Add weights to matrix
                for (idx1, idx2, weight) in [(i, j, 0.5 * cot_k), 
                                           (i, k, 0.5 * cot_j),
                                           (j, i, 0.5 * cot_k),
                                           (j, k, 0.5 * cot_i), 
                                           (k, i, 0.5 * cot_j),
                                           (k, j, 0.5 * cot_i)]:
                    I.append(idx1)
                    J.append(idx2) 
                    V.append(weight)
        
        # Create sparse matrix
        L = sparse.csr_matrix((V, (I, J)), shape=(n_vertices, n_vertices))
        
        # Set diagonal to negative sum of row
        row_sum = np.array(L.sum(axis=1)).flatten()
        L = L - sparse.diags(row_sum)
        
        return L
    
    def parameterize_boundary(self, boundary_vertices):
        """Parameterize boundary vertices to unit circle"""
        n_boundary = len(boundary_vertices)
        boundary_uv = np.zeros((n_boundary, 2))
        
        # Map boundary to unit circle
        for i, vertex_idx in enumerate(boundary_vertices):
            angle = 2 * np.pi * i / n_boundary
            boundary_uv[i] = [np.cos(angle), np.sin(angle)]
        
        return boundary_uv
    
    def solve_laplace_equation(self, L, boundary_vertices, boundary_uv):
        """Solve Laplace equation for interior vertices"""
        n_vertices = len(self.vertices)
        
        # Create mask for interior vertices
        interior_mask = np.ones(n_vertices, dtype=bool)
        interior_mask[boundary_vertices] = False
        interior_indices = np.where(interior_mask)[0]
        
        # Split matrix into interior and boundary parts
        L_ii = L[interior_indices, :][:, interior_indices]
        L_ib = L[interior_indices, :][:, boundary_vertices]
        
        # Solve for x and y coordinates separately
        uv = np.zeros((n_vertices, 2))
        uv[boundary_vertices] = boundary_uv
        
        # Right-hand side
        b_x = -L_ib.dot(boundary_uv[:, 0])
        b_y = -L_ib.dot(boundary_uv[:, 1])
        
        # Solve linear systems
        uv[interior_indices, 0] = spsolve(L_ii, b_x)
        uv[interior_indices, 1] = spsolve(L_ii, b_y)
        
        return uv
    
    def layout(self):
        """Compute the conformal map"""
        # Find boundary vertices
        boundary_vertices = self.find_boundary_vertices()
        
        # Compute cotangent Laplacian
        L = self.compute_cotangent_weights()
        
        # Parameterize boundary
        boundary_uv = self.parameterize_boundary(boundary_vertices)
        
        # Solve for interior vertices
        uv_vertices = self.solve_laplace_equation(L, boundary_vertices, boundary_uv)
        
        # UV faces are the same as original faces
        self.uv_vertices = uv_vertices
        self.uv_faces = self.faces
        
        return self

class MinimalSCP:
    """Minimal implementation of Spectral Conformal Parameterization"""
    
    def __init__(self, vertices, faces):
        self.vertices = vertices.copy()
        self.faces = faces
        self.uv_vertices = None
        self.uv_faces = None
        
    def layout(self):
        """Simple conformal parameterization - fallback to BFF for now"""
        # For simplicity, use BFF method
        bff = MinimalBFF(self.vertices, self.faces)
        result = bff.layout()
        self.uv_vertices = result.uv_vertices
        self.uv_faces = result.uv_faces
        return self

class MinimalAE:
    """Minimal implementation of Authalic Embedding"""
    
    def __init__(self, vertices, faces):
        self.vertices = vertices.copy()
        self.faces = faces
        self.uv_vertices = None
        self.uv_faces = None
        
    def layout(self):
        """Simple authalic parameterization - fallback to BFF for now"""
        # For simplicity, use BFF method
        bff = MinimalBFF(self.vertices, self.faces)
        result = bff.layout()
        self.uv_vertices = result.uv_vertices
        self.uv_faces = result.uv_faces
        return self

# Aliases for compatibility with original confmap API
BFF = MinimalBFF
SCP = MinimalSCP  
AE = MinimalAE
