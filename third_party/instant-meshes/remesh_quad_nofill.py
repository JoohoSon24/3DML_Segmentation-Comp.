#!/usr/bin/env python3
"""
Remesh a GLB file using Instant Meshes (quad-dominant), preserving vertex count and color.
No hole filling preprocessing.
Usage: python remesh_quad_nofill.py <input.glb> [output.glb] [target_vertices]
"""
import sys
import os
import subprocess
import numpy as np
import trimesh
from scipy.spatial import KDTree

INSTANT_MESHES = os.path.join(os.path.dirname(__file__), "instant-meshes", "Instant Meshes")


def load_mesh(glb_path):
    m = trimesh.load(glb_path)
    return list(m.geometry.values())[0]


def glb_to_obj(glb_path, obj_path):
    geom = load_mesh(glb_path)
    geom.export(obj_path)
    return len(geom.vertices)


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
        img = np.array(tex.convert("RGBA"))
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

    print("Step 1: Exporting OBJ...")
    n_vertices = glb_to_obj(input_glb, input_obj)
    if target_vertices is None:
        target_vertices = n_vertices
    print(f"Original vertices: {n_vertices}, target: {target_vertices}")

    im_target = max(100, target_vertices // 4)
    cmd = [INSTANT_MESHES, input_obj, "-o", output_obj, "--vertices", str(im_target)]
    print(f"Step 2: Running Instant Meshes (target={im_target})...")
    subprocess.run(cmd, check=True)

    print("Step 3: Transferring vertex colors from original...")
    original = load_mesh(input_glb)
    remeshed = trimesh.load(output_obj)
    remeshed = transfer_vertex_colors(original, remeshed)

    remeshed.export(output_glb)
    print(f"Done. Output: {output_glb} (V={len(remeshed.vertices)}, F={len(remeshed.faces)})")

    os.remove(input_obj)
    os.remove(output_obj)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    input_glb = sys.argv[1]
    output_glb = sys.argv[2] if len(sys.argv) > 2 else input_glb.replace(".glb", "_quad_nofill.glb")
    target_vertices = int(sys.argv[3]) if len(sys.argv) > 3 else None
    remesh(input_glb, output_glb, target_vertices)
