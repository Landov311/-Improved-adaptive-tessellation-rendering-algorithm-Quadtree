# Quadtree Structure in Adaptive Tessellation Rendering

This document summarizes the quadtree-based data structure and related logic described in the paper *Improved Adaptive Tessellation Rendering Algorithm* by Wang et al. (2023). It is intended to serve as a design reference for implementing a simplified MVP of the quadtree component using a code editor like Cursor.

## Overview

The paper describes an **adaptive tessellation rendering** algorithm with two main stages:

- **Offline stage**: Reads mesh data and constructs a subdivision hierarchy using a quadtree.
- **Online stage**: Uses the quadtree to efficiently evaluate surface patches in real time using shaders.

The **quadtree** stores information about the subdivision of surface patches and references control point templates.

## Quadtree Node Types

Each node in the quadtree represents a region of a mesh face. Leaf nodes represent subdomains that can be evaluated.

- `RegularNode`: Normal patch with 16 control points.
- `CreaseNode`: Patch with semi-sharp creases, stores crease sharpness.
- `SpecialNode`: For corners with unconventional topology (extreme vertex + tangents).
- `TerminalNode`: Stops further subdivision; stores 24 control points in a 5x5 mesh and a rotation index.

## Node Structure

### Internal Node
- Contains 4 children (standard quadtree structure).
- Each level represents a recursive subdivision step.

### Leaf Node (One of the following)
- `RegularNode`: Contains 16 control points.
- `CreaseNode`: Similar to regular but includes crease sharpness metadata.
- `SpecialNode`: 3 templates for evaluating a corner patch.
- `TerminalNode`: Dense control grid, no further subdivision.

## Templates and Control Points

- Templates encode control points as **weighted sums of one-ring vertices**.
- Each face maintains an ordered list of one-ring vertices.
- A **weight matrix** maps vertex indices to control point weights.
- This structure allows reuse and efficiency during runtime evaluation.

## Evaluation Flow (Simplified)

1. **Offline**
   - Build quadtree based on Catmull-Clark subdivision.
   - Attach templates and weight references to nodes.

2. **Online**
   - Traverse quadtree to find the evaluation node based on UV coordinates.
   - Evaluate surface patch using templates and control point weights.
   - For terminal/special nodes, use fixed logic for tangent/position interpolation.

## Algorithms Mentioned

- **Greedy traversal** through the quadtree to optimize for SIMT divergence.
- **Weight adjacency matrix** for ordering sensitive floating-point summation.
- **Fibonacci heap** (conceptually) for theoretical runtime optimization in traversal (optional for MVP).

## Suggested Implementation Plan

1. Define the `QuadtreeNode` class and its types (enum or subclasses).
2. Define the `Template` structure (list of weights + vertex indices).
3. Build a simple mesh and recursively subdivide it to form the tree.
4. Implement `insert`, `subdivide`, `query` methods.
5. Write unit tests for:
   - Correct template generation.
   - Subdivision depth.
   - Query accuracy at edges, centers, and corners.

## References

Wang, M. et al. (2023). *Improved Adaptive Tessellation Rendering Algorithm*. Technology and Health Care, 31(S81–S95). DOI: 10.3233/THC-236009
