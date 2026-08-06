import numpy as np

try:
    import pyopencl as cl
    HAS_PYOPENCL = True
except ImportError:
    HAS_PYOPENCL = False

CL_KERNEL_SOURCE = """
__kernel void compute_face_points(
    __global const float* vertices,
    __global const int* face_vertices,
    __global const int* face_offsets,
    __global float* face_points,
    const int num_faces)
{
    int i = get_global_id(0);
    if (i >= num_faces) return;
    
    int start = face_offsets[i];
    int end = face_offsets[i+1];
    int count = end - start;
    
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    for (int j = start; j < end; ++j) {
        int v_idx = face_vertices[j];
        fx += vertices[v_idx * 3 + 0];
        fy += vertices[v_idx * 3 + 1];
        fz += vertices[v_idx * 3 + 2];
    }
    
    if (count > 0) {
        face_points[i * 3 + 0] = fx / count;
        face_points[i * 3 + 1] = fy / count;
        face_points[i * 3 + 2] = fz / count;
    }
}

__kernel void compute_edge_points(
    __global const float* vertices,
    __global const float* face_points,
    __global const int* edge_vertices,
    __global const int* edge_faces,
    __global float* edge_points,
    const int num_edges)
{
    int i = get_global_id(0);
    if (i >= num_edges) return;
    
    int v0 = edge_vertices[i * 2 + 0];
    int v1 = edge_vertices[i * 2 + 1];
    int f0 = edge_faces[i * 2 + 0];
    int f1 = edge_faces[i * 2 + 1];
    
    float v0x = vertices[v0 * 3 + 0], v0y = vertices[v0 * 3 + 1], v0z = vertices[v0 * 3 + 2];
    float v1x = vertices[v1 * 3 + 0], v1y = vertices[v1 * 3 + 1], v1z = vertices[v1 * 3 + 2];
    
    if (f1 != -1) {
        // Interior edge
        float f0x = face_points[f0 * 3 + 0], f0y = face_points[f0 * 3 + 1], f0z = face_points[f0 * 3 + 2];
        float f1x = face_points[f1 * 3 + 0], f1y = face_points[f1 * 3 + 1], f1z = face_points[f1 * 3 + 2];
        
        edge_points[i * 3 + 0] = (v0x + v1x + f0x + f1x) / 4.0f;
        edge_points[i * 3 + 1] = (v0y + v1y + f0y + f1y) / 4.0f;
        edge_points[i * 3 + 2] = (v0z + v1z + f0z + f1z) / 4.0f;
    } else {
        // Boundary edge
        edge_points[i * 3 + 0] = (v0x + v1x) / 2.0f;
        edge_points[i * 3 + 1] = (v0y + v1y) / 2.0f;
        edge_points[i * 3 + 2] = (v0z + v1z) / 2.0f;
    }
}

__kernel void compute_vertex_points(
    __global const float* vertices,
    __global const float* face_points,
    __global const int* edge_vertices,
    __global const int* edge_faces,
    __global const int* vertex_faces,
    __global const int* vertex_face_offsets,
    __global const int* vertex_edges,
    __global const int* vertex_edge_offsets,
    __global float* vertex_points_out,
    const int num_vertices)
{
    int i = get_global_id(0);
    if (i >= num_vertices) return;
    
    float px = vertices[i * 3 + 0];
    float py = vertices[i * 3 + 1];
    float pz = vertices[i * 3 + 2];
    
    int e_start = vertex_edge_offsets[i];
    int e_end = vertex_edge_offsets[i+1];
    int n_edges = e_end - e_start;
    
    int f_start = vertex_face_offsets[i];
    int f_end = vertex_face_offsets[i+1];
    int n_faces = f_end - f_start;
    
    int bound_count = 0;
    float bx = 0.0f, by = 0.0f, bz = 0.0f;
    
    for (int j = e_start; j < e_end; ++j) {
        int e_idx = vertex_edges[j];
        if (edge_faces[e_idx * 2 + 1] == -1) {
            bound_count++;
            int v0 = edge_vertices[e_idx * 2 + 0];
            int v1 = edge_vertices[e_idx * 2 + 1];
            float mx = (vertices[v0 * 3 + 0] + vertices[v1 * 3 + 0]) / 2.0f;
            float my = (vertices[v0 * 3 + 1] + vertices[v1 * 3 + 1]) / 2.0f;
            float mz = (vertices[v0 * 3 + 2] + vertices[v1 * 3 + 2]) / 2.0f;
            bx += mx;
            by += my;
            bz += mz;
        }
    }
    
    if (bound_count > 0) {
        // Boundary vertex rule
        bx /= bound_count;
        by /= bound_count;
        bz /= bound_count;
        
        vertex_points_out[i * 3 + 0] = (bx + px) / 2.0f;
        vertex_points_out[i * 3 + 1] = (by + py) / 2.0f;
        vertex_points_out[i * 3 + 2] = (bz + pz) / 2.0f;
    } else {
        // Interior vertex rule
        float Qx = 0.0f, Qy = 0.0f, Qz = 0.0f;
        for (int j = f_start; j < f_end; ++j) {
            int f_idx = vertex_faces[j];
            Qx += face_points[f_idx * 3 + 0];
            Qy += face_points[f_idx * 3 + 1];
            Qz += face_points[f_idx * 3 + 2];
        }
        if (n_faces > 0) {
            Qx /= n_faces; Qy /= n_faces; Qz /= n_faces;
        }
        
        float Rx = 0.0f, Ry = 0.0f, Rz = 0.0f;
        for (int j = e_start; j < e_end; ++j) {
            int e_idx = vertex_edges[j];
            int v0 = edge_vertices[e_idx * 2 + 0];
            int v1 = edge_vertices[e_idx * 2 + 1];
            Rx += (vertices[v0 * 3 + 0] + vertices[v1 * 3 + 0]) / 2.0f;
            Ry += (vertices[v0 * 3 + 1] + vertices[v1 * 3 + 1]) / 2.0f;
            Rz += (vertices[v0 * 3 + 2] + vertices[v1 * 3 + 2]) / 2.0f;
        }
        if (n_edges > 0) {
            Rx /= n_edges; Ry /= n_edges; Rz /= n_edges;
        }
        
        float n = (float)n_edges;
        if (n > 0) {
            vertex_points_out[i * 3 + 0] = (Qx + 2.0f * Rx + (n - 3.0f) * px) / n;
            vertex_points_out[i * 3 + 1] = (Qy + 2.0f * Ry + (n - 3.0f) * py) / n;
            vertex_points_out[i * 3 + 2] = (Qz + 2.0f * Rz + (n - 3.0f) * pz) / n;
        } else {
            vertex_points_out[i * 3 + 0] = px;
            vertex_points_out[i * 3 + 1] = py;
            vertex_points_out[i * 3 + 2] = pz;
        }
    }
}
__kernel void compute_limit_surface(
    __global const float* vertices,
    __global const float* face_points,
    __global const int* edge_vertices,
    __global const int* edge_faces,
    __global const int* vertex_faces,
    __global const int* vertex_face_offsets,
    __global const int* vertex_edges,
    __global const int* vertex_edge_offsets,
    __global float* limit_points_out,
    const int num_vertices)
{
    int i = get_global_id(0);
    if (i >= num_vertices) return;
    
    float px = vertices[i * 3 + 0];
    float py = vertices[i * 3 + 1];
    float pz = vertices[i * 3 + 2];
    
    int e_start = vertex_edge_offsets[i];
    int e_end = vertex_edge_offsets[i+1];
    int n_edges = e_end - e_start;
    
    int f_start = vertex_face_offsets[i];
    int f_end = vertex_face_offsets[i+1];
    int n_faces = f_end - f_start;
    
    bool is_boundary = false;
    for (int j = e_start; j < e_end; ++j) {
        int e_idx = vertex_edges[j];
        if (edge_faces[e_idx * 2 + 1] == -1) {
            is_boundary = true;
            break;
        }
    }
    
    if (is_boundary || n_edges == 0) {
        limit_points_out[i * 3 + 0] = px;
        limit_points_out[i * 3 + 1] = py;
        limit_points_out[i * 3 + 2] = pz;
    } else {
        float sum_Fx = 0.0f, sum_Fy = 0.0f, sum_Fz = 0.0f;
        for (int j = f_start; j < f_end; ++j) {
            int f_idx = vertex_faces[j];
            sum_Fx += face_points[f_idx * 3 + 0];
            sum_Fy += face_points[f_idx * 3 + 1];
            sum_Fz += face_points[f_idx * 3 + 2];
        }
        
        float sum_Rx = 0.0f, sum_Ry = 0.0f, sum_Rz = 0.0f;
        for (int j = e_start; j < e_end; ++j) {
            int e_idx = vertex_edges[j];
            int v0 = edge_vertices[e_idx * 2 + 0];
            int v1 = edge_vertices[e_idx * 2 + 1];
            sum_Rx += (vertices[v0 * 3 + 0] + vertices[v1 * 3 + 0]) / 2.0f;
            sum_Ry += (vertices[v0 * 3 + 1] + vertices[v1 * 3 + 1]) / 2.0f;
            sum_Rz += (vertices[v0 * 3 + 2] + vertices[v1 * 3 + 2]) / 2.0f;
        }
        
        float n = (float)n_edges;
        float factor_P = (n - 2.0f) / n;
        float factor_RF = 1.0f / (n * n);
        
        limit_points_out[i * 3 + 0] = factor_P * px + factor_RF * sum_Rx + factor_RF * sum_Fx;
        limit_points_out[i * 3 + 1] = factor_P * py + factor_RF * sum_Ry + factor_RF * sum_Fy;
        limit_points_out[i * 3 + 2] = factor_P * pz + factor_RF * sum_Rz + factor_RF * sum_Fz;
    }
}
"""

class OpenCLSubdivider:
    def __init__(self):
        if not HAS_PYOPENCL:
            raise RuntimeError("pyopencl is not installed or available.")
        
        platforms = cl.get_platforms()
        device = None
        for p in platforms:
            for d in p.get_devices():
                if d.type == cl.device_type.GPU:
                    device = d
                    break
            if device:
                break
        if not device:
            device = platforms[0].get_devices()[0]
            
        self.ctx = cl.Context([device])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, CL_KERNEL_SOURCE).build()

    def _extract_topology(self, num_vertices, face_vertices, face_offsets):
        num_faces = len(face_offsets) - 1
        
        edges_dict = {}
        vertex_faces_list = [[] for _ in range(num_vertices)]
        vertex_edges_list = [[] for _ in range(num_vertices)]
        
        for f in range(num_faces):
            start = face_offsets[f]
            end = face_offsets[f+1]
            count = end - start
            
            for i in range(count):
                v0 = face_vertices[start + i]
                v1 = face_vertices[start + (i + 1) % count]
                
                vertex_faces_list[v0].append(f)
                
                e_key = (min(v0, v1), max(v0, v1))
                if e_key not in edges_dict:
                    edges_dict[e_key] = {
                        'index': len(edges_dict),
                        'v0': v0,
                        'v1': v1,
                        'faces': []
                    }
                edges_dict[e_key]['faces'].append(f)

        num_edges = len(edges_dict)
        edge_vertices = np.empty((num_edges, 2), dtype=np.int32)
        edge_faces = np.full((num_edges, 2), -1, dtype=np.int32)
        
        edge_map = {}
        for e_key, e_data in edges_dict.items():
            idx = e_data['index']
            edge_map[e_key] = idx
            edge_vertices[idx, 0] = e_data['v0']
            edge_vertices[idx, 1] = e_data['v1']
            
            f_list = e_data['faces']
            edge_faces[idx, 0] = f_list[0]
            if len(f_list) > 1:
                edge_faces[idx, 1] = f_list[1]
                
            vertex_edges_list[e_data['v0']].append(idx)
            vertex_edges_list[e_data['v1']].append(idx)
            
        vertex_face_offsets = np.zeros(num_vertices + 1, dtype=np.int32)
        vertex_edge_offsets = np.zeros(num_vertices + 1, dtype=np.int32)
        
        vf_count = sum(len(vf) for vf in vertex_faces_list)
        ve_count = sum(len(ve) for ve in vertex_edges_list)
        
        vertex_faces = np.empty(vf_count, dtype=np.int32)
        vertex_edges = np.empty(ve_count, dtype=np.int32)
        
        vf_idx = 0
        ve_idx = 0
        for v in range(num_vertices):
            vertex_face_offsets[v] = vf_idx
            for f in vertex_faces_list[v]:
                vertex_faces[vf_idx] = f
                vf_idx += 1
                
            vertex_edge_offsets[v] = ve_idx
            for e in vertex_edges_list[v]:
                vertex_edges[ve_idx] = e
                ve_idx += 1
                
        vertex_face_offsets[num_vertices] = vf_idx
        vertex_edge_offsets[num_vertices] = ve_idx
        
        return (edge_vertices, edge_faces, 
                vertex_faces, vertex_face_offsets, 
                vertex_edges, vertex_edge_offsets, edge_map)

    def subdivide(self, vertices, face_vertices, face_offsets=None):
        """
        Subdivide a mesh using OpenCL kernels.
        
        Args:
            vertices: (V, 3) float32 numpy array
            face_vertices: 1D int32 array of face indices or 2D array of (F, N)
            face_offsets: 1D int32 array of offsets, required if face_vertices is 1D
            
        Returns:
            new_vertices: Float32 array of the subdivided vertices
            new_face_vertices: Int32 flat array of new faces (always quads)
            new_face_offsets: Int32 array of face offsets for new_face_vertices
        """
        if face_offsets is None:
            # Assume face_vertices is 2D array of uniform faces (e.g. Quads)
            faces_2d = np.asarray(face_vertices)
            num_faces, n_verts = faces_2d.shape
            face_vertices = faces_2d.flatten().astype(np.int32)
            face_offsets = np.arange(0, (num_faces + 1) * n_verts, n_verts, dtype=np.int32)
        else:
            face_vertices = np.asarray(face_vertices, dtype=np.int32)
            face_offsets = np.asarray(face_offsets, dtype=np.int32)
            
        vertices = np.asarray(vertices, dtype=np.float32)
        num_vertices = len(vertices)
        num_faces = len(face_offsets) - 1
        
        # 1. CPU topology extraction
        (edge_vertices, edge_faces, 
         vertex_faces, vertex_face_offsets, 
         vertex_edges, vertex_edge_offsets, edge_map) = self._extract_topology(
            num_vertices, face_vertices, face_offsets
        )
        
        num_edges = len(edge_vertices)
        
        # 2. OpenCL Buffers setup
        mf = cl.mem_flags
        
        d_vertices = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vertices)
        d_face_vertices = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=face_vertices)
        d_face_offsets = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=face_offsets)
        
        d_edge_vertices = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=edge_vertices)
        d_edge_faces = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=edge_faces)
        
        d_vertex_faces = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vertex_faces)
        d_vertex_face_offsets = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vertex_face_offsets)
        d_vertex_edges = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vertex_edges)
        d_vertex_edge_offsets = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vertex_edge_offsets)
        
        # Outputs
        face_points = np.empty((num_faces, 3), dtype=np.float32)
        edge_points = np.empty((num_edges, 3), dtype=np.float32)
        vertex_points = np.empty((num_vertices, 3), dtype=np.float32)
        
        d_face_points = cl.Buffer(self.ctx, mf.READ_WRITE, face_points.nbytes)
        d_edge_points = cl.Buffer(self.ctx, mf.READ_WRITE, edge_points.nbytes)
        d_vertex_points = cl.Buffer(self.ctx, mf.READ_WRITE, vertex_points.nbytes)
        
        # 3. Kernel Executions
        self.prg.compute_face_points(
            self.queue, (num_faces,), None,
            d_vertices, d_face_vertices, d_face_offsets, d_face_points, np.int32(num_faces)
        )
        
        self.prg.compute_edge_points(
            self.queue, (num_edges,), None,
            d_vertices, d_face_points, d_edge_vertices, d_edge_faces, d_edge_points, np.int32(num_edges)
        )
        
        self.prg.compute_vertex_points(
            self.queue, (num_vertices,), None,
            d_vertices, d_face_points, d_edge_vertices, d_edge_faces,
            d_vertex_faces, d_vertex_face_offsets, d_vertex_edges, d_vertex_edge_offsets,
            d_vertex_points, np.int32(num_vertices)
        )
        
        # Read back results
        cl.enqueue_copy(self.queue, face_points, d_face_points)
        cl.enqueue_copy(self.queue, edge_points, d_edge_points)
        cl.enqueue_copy(self.queue, vertex_points, d_vertex_points)
        
        self.queue.finish()
        
        # 4. Build new faces
        total_new_quads = len(face_vertices)
        new_face_vertices = np.empty(total_new_quads * 4, dtype=np.int32)
        
        quad_idx = 0
        for f in range(num_faces):
            start = face_offsets[f]
            end = face_offsets[f+1]
            count = end - start
            
            f_pt = num_vertices + num_edges + f
            
            for i in range(count):
                v_curr = face_vertices[start + i]
                v_next = face_vertices[start + (i + 1) % count]
                v_prev = face_vertices[start + (i - 1 + count) % count]
                
                e_next_key = (min(v_curr, v_next), max(v_curr, v_next))
                e_prev_key = (min(v_curr, v_prev), max(v_curr, v_prev))
                
                e_next_idx = edge_map[e_next_key]
                e_prev_idx = edge_map[e_prev_key]
                
                e_next_pt = num_vertices + e_next_idx
                e_prev_pt = num_vertices + e_prev_idx
                
                # Counter-clockwise quad matching standard Catmull-Clark
                new_face_vertices[quad_idx * 4 + 0] = v_curr
                new_face_vertices[quad_idx * 4 + 1] = e_next_pt
                new_face_vertices[quad_idx * 4 + 2] = f_pt
                new_face_vertices[quad_idx * 4 + 3] = e_prev_pt
                
                quad_idx += 1
                
        new_face_offsets = np.arange(0, total_new_quads * 4 + 1, 4, dtype=np.int32)
        
        # Combine into final vertex array
        new_vertices = np.vstack([vertex_points, edge_points, face_points])
        
        return new_vertices, new_face_vertices, new_face_offsets

    def evaluate_limit_surface(self, vertices, face_vertices, face_offsets=None):
        """
        Evaluate limit surface positions using OpenCL kernel.
        """
        if face_offsets is None:
            faces_2d = np.asarray(face_vertices)
            num_faces, n_verts = faces_2d.shape
            face_vertices = faces_2d.flatten().astype(np.int32)
            face_offsets = np.arange(0, (num_faces + 1) * n_verts, n_verts, dtype=np.int32)
        else:
            face_vertices = np.asarray(face_vertices, dtype=np.int32)
            face_offsets = np.asarray(face_offsets, dtype=np.int32)
            
        vertices = np.asarray(vertices, dtype=np.float32)
        num_vertices = len(vertices)
        num_faces = len(face_offsets) - 1
        
        (edge_vertices, edge_faces, 
         vertex_faces, vertex_face_offsets, 
         vertex_edges, vertex_edge_offsets, edge_map) = self._extract_topology(
            num_vertices, face_vertices, face_offsets
        )
        
        mf = cl.mem_flags
        
        d_vertices = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vertices)
        d_face_vertices = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=face_vertices)
        d_face_offsets = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=face_offsets)
        
        d_edge_vertices = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=edge_vertices)
        d_edge_faces = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=edge_faces)
        
        d_vertex_faces = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vertex_faces)
        d_vertex_face_offsets = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vertex_face_offsets)
        d_vertex_edges = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vertex_edges)
        d_vertex_edge_offsets = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vertex_edge_offsets)
        
        face_points = np.empty((num_faces, 3), dtype=np.float32)
        limit_points = np.empty((num_vertices, 3), dtype=np.float32)
        
        d_face_points = cl.Buffer(self.ctx, mf.READ_WRITE, face_points.nbytes)
        d_limit_points = cl.Buffer(self.ctx, mf.READ_WRITE, limit_points.nbytes)
        
        self.prg.compute_face_points(
            self.queue, (num_faces,), None,
            d_vertices, d_face_vertices, d_face_offsets, d_face_points, np.int32(num_faces)
        )
        
        self.prg.compute_limit_surface(
            self.queue, (num_vertices,), None,
            d_vertices, d_face_points, d_edge_vertices, d_edge_faces,
            d_vertex_faces, d_vertex_face_offsets, d_vertex_edges, d_vertex_edge_offsets,
            d_limit_points, np.int32(num_vertices)
        )
        
        cl.enqueue_copy(self.queue, limit_points, d_limit_points)
        self.queue.finish()
        
        return limit_points
