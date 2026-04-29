#!/usr/bin/env python3
"""
Remesh a GLB file using Poisson reconstruction + decimation + Instant Meshes (quad).
Steps: Poisson surface reconstruction → decimate to original face count → Instant Meshes quad remesh → transfer vertex colors
Usage: python remesh_quad_poisson.py <input.glb> [output.glb] [target_vertices]
"""
import sys
import os
import subprocess
import numpy as np
import trimesh
import pymeshlab
from scipy.spatial import KDTree

INSTANT_MESHES = os.path.join(os.path.dirname(__file__), "instant-meshes", "Instant Meshes")


def load_mesh(glb_path):
    m = trimesh.load(glb_path)
    return list(m.geometry.values())[0]


def transfer_vertex_colors(original_mesh, new_mesh):
    tree = KDTree(original_mesh.vertices)
    _, idx = tree.query(new_mesh.vertices)

    if original_mesh.visual.kind == 'texture':
        uv = original_mesh.visual.uv
        mat = original_mesh.visual.material
        tex = getattr(mat, 'baseColorTexture', None) or getattr(mat, 'image', None)
        if tex is None:
            print("Warning: no texture image found, skipping color transfer")
            return new_mesh
        img = np.array(tex.convert('RGBA'))
        h, w = img.shape[:2]
        u = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
        v = np.clip(((1 - uv[:, 1]) * h).astype(int), 0, h - 1)
        orig_colors = img[v, u]
    elif hasattr(original_mesh.visual, 'vertex_colors'):
        orig_colors = original_mesh.visual.vertex_colors
    else:
        print("Warning: no color info found on original mesh")
        return new_mesh

    new_mesh.visual = trimesh.visual.ColorVisuals(mesh=new_mesh, vertex_colors=orig_colors[idx])
    return new_mesh


def remesh(input_glb, output_glb, target_vertices=None):
    base = os.path.splitext(input_glb)[0]
    input_obj = base + "_input.obj"
    output_obj = base + "_remeshed.obj"

    print("Step 1: Poisson reconstruction + decimation...")
    orig = load_mesh(input_glb)
    orig.export(input_obj)
    n_vertices = len(orig.vertices)
    n_faces = len(orig.faces)
    if target_vertices is None:
        target_vertices = n_vertices
    print(f"Original: {n_vertices} vertices, {n_faces} faces, target: {target_vertices}")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_obj)
    ms.generate_surface_reconstruction_screened_poisson(depth=8)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=n_faces)
    print(f"After Poisson+decimate: {ms.current_mesh().vertex_number()} vertices, {ms.current_mesh().face_number()} faces")

    m = ms.current_mesh()
    interim = trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix(), process=False)
    interim.export(input_obj)

    im_target = max(100, target_vertices // 4)
    cmd = [INSTANT_MESHES, input_obj, "-o", output_obj, "--vertices", str(im_target)]
    print(f"Step 2: Running Instant Meshes (target={im_target})...")
    subprocess.run(cmd, check=True)

    print("Step 3: Transferring vertex colors from original...")
    remeshed = trimesh.load(output_obj)
    if hasattr(remeshed, 'geometry'):
        remeshed = list(remeshed.geometry.values())[0]
    remeshed = transfer_vertex_colors(orig, remeshed)

    remeshed.export(output_glb)
    print(f"Done. Output: {output_glb} (V={len(remeshed.vertices)}, F={len(remeshed.faces)})")

    os.remove(input_obj)
    os.remove(output_obj)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    input_glb = sys.argv[1]
    output_glb = sys.argv[2] if len(sys.argv) > 2 else input_glb.replace(".glb", "_quad_poisson.glb")
    target_vertices = int(sys.argv[3]) if len(sys.argv) > 3 else None
    remesh(input_glb, output_glb, target_vertices)
