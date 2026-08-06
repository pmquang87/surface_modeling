# Advanced Surface Modeling Features

This document provides visual tutorials and documentation for the advanced features introduced in our surface modeling toolkit: GPU Subdivision, T-Splines, and G3 NURBS.

## 1. GPU Subdivision

GPU Subdivision offloads the computational heavy lifting of surface refinement to the graphics processing unit. This allows for real-time visualization of high-density meshes.

### Key Benefits
*   **Real-time Performance:** Manipulate base meshes while seeing the subdivided limit surface instantly.
*   **Memory Efficiency:** Subdivision happens on the fly in the GPU pipeline (e.g., using tessellation shaders), saving host memory.
*   **Adaptive Tessellation:** Dynamic level of detail based on camera distance or surface curvature.

## 2. T-Splines

T-Splines are an evolution of NURBS that allow for local refinement without the need for control points to propagate across the entire surface. This is achieved by allowing T-junctions in the control grid.

### T-Spline Topology

The following diagram illustrates the difference between standard NURBS and T-Spline topology:

```mermaid
graph TD
    subgraph "Standard NURBS"
        N1(( )) --- N2(( )) --- N3(( ))
        N4(( )) --- N5(( )) --- N6(( ))
        N7(( )) --- N8(( )) --- N9(( ))
        
        N1 --- N4 --- N7
        N2 --- N5 --- N8
        N3 --- N6 --- N9
    end

    subgraph "T-Spline Topology"
        T1(( )) --- T2(( )) --- T3(( )) --- T4(( ))
        T5(( )) --- T6(( )) --- T7(( )) --- T8(( ))
        
        T1 --- T5
        T2 --- T6
        
        %% T-Junction here
        TJ((T-Junction))
        T3 --- TJ
        TJ --- T7
        
        T4 --- T8
        
        %% Local refinement inserted without full propagation
        L1((Local)) --- TJ --- L2((Local))
        
        style TJ fill:#f9f,stroke:#333,stroke-width:4px
    end
```

*In a standard NURBS grid, inserting a control point adds an entire row/column. In a T-Spline, a T-junction allows a partial row/column, drastically reducing the number of unnecessary control points.*

## 3. G3 NURBS (Curvature Acceleration Continuity)

G3 continuity goes beyond the standard G2 (curvature continuity) by ensuring that the *rate of change of curvature* (curvature acceleration) is also continuous across a surface boundary. This is critical for Class-A surfacing in automotive and aerospace design, as it prevents any visible "kinks" in reflections.

### G3 Continuity Visualized

```mermaid
graph LR
    subgraph "Continuity Levels"
        direction TB
        G0[G0 - Position: Surfaces touch]
        G1[G1 - Tangency: Same direction]
        G2[G2 - Curvature: Same radius]
        G3[G3 - Acceleration: Same rate of curvature change]
        
        G0 --> G1 --> G2 --> G3
    end

    subgraph "G3 Boundary Matching"
        direction LR
        S1[Surface 1] -- "P (Position)" --> B[Boundary]
        S2[Surface 2] -- "P (Position)" --> B
        
        S1 -- "V (Tangent Vector)" --> B
        S2 -- "V (Tangent Vector)" --> B
        
        S1 -- "K (Curvature)" --> B
        S2 -- "K (Curvature)" --> B
        
        S1 -- "dK/ds (Rate of Curvature)" --> B
        S2 -- "dK/ds (Rate of Curvature)" --> B
    end
    
    style G3 fill:#bbf,stroke:#333,stroke-width:2px
```

Achieving G3 requires matching up to the 3rd derivative across the boundary, often requiring higher-degree NURBS curves (degree 7 or 9) and precise alignment of the first four rows of control points on either side of the boundary.
