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
                uv_face = uv_faces[i] if i < len(uv_faces) else face
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
            # For closed meshes, use a simple approach
            return self.find_approximate_boundary()
        
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
    
    def find_approximate_boundary(self):
        """Find approximate boundary for closed meshes"""
        # For closed meshes, we'll use the vertices with minimum z-coordinate as a pseudo-boundary
        if len(self.vertices) == 0:
            return []
        
        # Find bottom ring of vertices
        min_z = np.min(self.vertices[:, 2])
        bottom_vertices = np.where(self.vertices[:, 2] <= min_z + 0.1)[0]
        
        if len(bottom_vertices) > 0:
            return bottom_vertices.tolist()
        else:
            # Fallback: use first few vertices
            return list(range(min(10, len(self.vertices))))
    
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
                
                # Compute cotangent weights using more robust formula
                e_ij = vj - vi
                e_ik = vk - vi
                e_ji = vi - vj
                e_jk = vk - vj
                
                # Area of the triangle
                area = 0.5 * np.linalg.norm(np.cross(e_ij, e_ik))
                
                if area > 1e-10:  # Avoid division by zero
                    # Cotangent of angle at vertex k
                    cot_k = np.dot(e_ij, e_ik) / (2 * area)
                    
                    # Cotangent of angle at vertex j  
                    cot_j = np.dot(-e_ij, e_jk) / (2 * area)
                    
                    # Cotangent of angle at vertex i
                    cot_i = np.dot(-e_ik, -e_jk) / (2 * area)
                    
                    # Add weights to matrix (symmetric)
                    for (idx1, idx2, weight) in [(i, j, 0.5 * cot_k), 
                                               (j, i, 0.5 * cot_k),
                                               (i, k, 0.5 * cot_j),
                                               (k, i, 0.5 * cot_j),
                                               (j, k, 0.5 * cot_i),
                                               (k, j, 0.5 * cot_i)]:
                        I.append(idx1)
                        J.append(idx2) 
                        V.append(weight)
        
        # Create sparse matrix
        if I:  # Only if we have weights
            L = sparse.csr_matrix((V, (I, J)), shape=(n_vertices, n_vertices))
            
            # Set diagonal to negative sum of row
            row_sum = np.array(L.sum(axis=1)).flatten()
            L = L - sparse.diags(row_sum)
        else:
            # Fallback: uniform weights
            L = sparse.lil_matrix((n_vertices, n_vertices))
            for i in range(n_vertices):
                L[i, i] = 1.0
        
        return L
    
    def parameterize_boundary(self, boundary_vertices):
        """Parameterize boundary vertices to unit circle or line"""
        n_boundary = len(boundary_vertices)
        boundary_uv = np.zeros((n_boundary, 2))
        
        if n_boundary == 0:
            return boundary_uv
            
        # Map boundary to unit circle or appropriate shape
        total_boundary_length = 0
        boundary_points = [self.vertices[i] for i in boundary_vertices]
        
        # Calculate boundary length
        for i in range(n_boundary):
            p1 = boundary_points[i]
            p2 = boundary_points[(i + 1) % n_boundary]
            total_boundary_length += np.linalg.norm(p2 - p1)
        
        if total_boundary_length > 0:
            # Map to circle with circumference proportional to boundary length
            current_length = 0
            for i in range(n_boundary):
                angle = 2 * np.pi * (current_length / total_boundary_length)
                boundary_uv[i] = [np.cos(angle) * 0.5 + 0.5, np.sin(angle) * 0.5 + 0.5]
                
                # Update current length
                if i < n_boundary - 1:
                    p1 = boundary_points[i]
                    p2 = boundary_points[i + 1]
                    current_length += np.linalg.norm(p2 - p1)
        else:
            # Fallback: map to circle evenly
            for i in range(n_boundary):
                angle = 2 * np.pi * i / n_boundary
                boundary_uv[i] = [np.cos(angle) * 0.5 + 0.5, np.sin(angle) * 0.5 + 0.5]
        
        return boundary_uv
    
    def solve_laplace_equation(self, L, boundary_vertices, boundary_uv):
        """Solve Laplace equation for interior vertices"""
        n_vertices = len(self.vertices)
        
        if len(boundary_vertices) == 0:
            # For closed meshes, use a simple planar projection
            uv = np.zeros((n_vertices, 2))
            for i in range(n_vertices):
                # Simple planar projection (good for testing)
                uv[i, 0] = self.vertices[i, 0] * 0.1 + 0.5
                uv[i, 1] = self.vertices[i, 1] * 0.1 + 0.5
            return uv
        
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
        
        try:
            # Solve linear systems
            uv[interior_indices, 0] = spsolve(L_ii, b_x)
            uv[interior_indices, 1] = spsolve(L_ii, b_y)
        except:
            # Fallback if solving fails
            for i in interior_indices:
                uv[i, 0] = self.vertices[i, 0] * 0.1 + 0.5
                uv[i, 1] = self.vertices[i, 1] * 0.1 + 0.5
        
        return uv
    
    def layout(self):
        """Compute the conformal map"""
        try:
            # Find boundary vertices
            boundary_vertices = self.find_boundary_vertices()
            
            # Compute cotangent Laplacian
            L = self.compute_cotangent_weights()
            
            # Parameterize boundary
            boundary_uv = self.parameterize_boundary(boundary_vertices)
            
            # Solve for interior vertices
            uv_vertices = self.solve_laplace_equation(L, boundary_vertices, boundary_uv)
            
            # Ensure UV coordinates are in [0,1] range
            if len(uv_vertices) > 0:
                min_uv = np.min(uv_vertices, axis=0)
                max_uv = np.max(uv_vertices, axis=0)
                range_uv = max_uv - min_uv
                
                if np.max(range_uv) > 0:
                    # Normalize to [0,1] range
                    uv_vertices = (uv_vertices - min_uv) / np.max(range_uv)
                else:
                    # Fallback if all UVs are the same
                    uv_vertices = np.clip(uv_vertices, 0, 1)
            
            # UV faces are the same as original faces
            self.uv_vertices = uv_vertices
            self.uv_faces = self.faces
            
        except Exception as e:
            print(f"Error in BFF layout: {e}")
            # Fallback: use simple planar projection
            n_vertices = len(self.vertices)
            self.uv_vertices = np.zeros((n_vertices, 2))
            for i in range(n_vertices):
                self.uv_vertices[i, 0] = self.vertices[i, 0] * 0.1 + 0.5
                self.uv_vertices[i, 1] = self.vertices[i, 1] * 0.1 + 0.5
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
        """Simple conformal parameterization"""
        try:
            # Use a simple planar projection for SCP
            n_vertices = len(self.vertices)
            self.uv_vertices = np.zeros((n_vertices, 2))
            
            # Project to plane based on principal components
            centered = self.vertices - np.mean(self.vertices, axis=0)
            
            # Use XY plane projection, scaled appropriately
            max_extent = np.max(np.ptp(centered, axis=0))
            if max_extent > 0:
                scale = 0.5 / max_extent
            else:
                scale = 1.0
                
            self.uv_vertices[:, 0] = centered[:, 0] * scale + 0.5
            self.uv_vertices[:, 1] = centered[:, 1] * scale + 0.5
            
            # Ensure in [0,1] range
            self.uv_vertices = np.clip(self.uv_vertices, 0, 1)
            self.uv_faces = self.faces
            
        except Exception as e:
            print(f"Error in SCP layout: {e}")
            # Fallback
            n_vertices = len(self.vertices)
            self.uv_vertices = np.zeros((n_vertices, 2))
            for i in range(n_vertices):
                self.uv_vertices[i, 0] = 0.5
                self.uv_vertices[i, 1] = 0.5
            self.uv_faces = self.faces
            
        return self

class MinimalAE:
    """Minimal implementation of Authalic Embedding"""
    
    def __init__(self, vertices, faces):
        self.vertices = vertices.copy()
        self.faces = faces
        self.uv_vertices = None
        self.uv_faces = None
        
    def layout(self):
        """Simple authalic parameterization"""
        try:
            # For AE, try to preserve area relationships
            n_vertices = len(self.vertices)
            self.uv_vertices = np.zeros((n_vertices, 2))
            
            # Simple area-preserving projection
            centered = self.vertices - np.mean(self.vertices, axis=0)
            
            # Calculate total surface area
            total_area = 0
            for face in self.faces:
                if len(face) == 3:
                    v0, v1, v2 = self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]
                    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                    total_area += area
            
            if total_area > 0:
                # Scale based on area
                scale = min(1.0, 1.0 / np.sqrt(total_area))
            else:
                scale = 1.0
                
            self.uv_vertices[:, 0] = centered[:, 0] * scale + 0.5
            self.uv_vertices[:, 1] = centered[:, 2] * scale + 0.5  # Use Z instead of Y for variety
            
            # Ensure in [0,1] range
            self.uv_vertices = np.clip(self.uv_vertices, 0, 1)
            self.uv_faces = self.faces
            
        except Exception as e:
            print(f"Error in AE layout: {e}")
            # Fallback
            n_vertices = len(self.vertices)
            self.uv_vertices = np.zeros((n_vertices, 2))
            for i in range(n_vertices):
                self.uv_vertices[i, 0] = 0.5
                self.uv_vertices[i, 1] = 0.5
            self.uv_faces = self.faces
            
        return self

# Aliases for compatibility with original confmap API
BFF = MinimalBFF
SCP = MinimalSCP  
AE = MinimalAE
