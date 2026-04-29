#!/usr/bin/env python3
"""
Remesh a GLB file using pymeshlab, preserving vertex count.
Usage: python remesh.py <input.glb> [output.glb]
"""
import sys
import os
import trimesh
import pymeshlab


def glb_to_obj(glb_path, obj_path):
    m = trimesh.load(glb_path)
    geom = list(m.geometry.values())[0]
    geom.export(obj_path)
    return len(geom.vertices)


def obj_to_glb(obj_path, glb_path):
    m = trimesh.load(obj_path)
    m.export(glb_path)


def remesh(input_glb, output_glb, target_vertices=None):
    base = os.path.splitext(input_glb)[0]
    input_obj = base + "_input.obj"
    output_obj = base + "_remeshed.obj"

    print(f"Converting {input_glb} -> {input_obj}")
    n_vertices = glb_to_obj(input_glb, input_obj)
    if target_vertices is None:
        target_vertices = n_vertices
    print(f"Original vertices: {n_vertices}, target: {target_vertices}")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_obj)

    # Estimate target edge length from vertex count
    current = ms.current_mesh()
    bbox = current.bounding_box()
    diag = bbox.diagonal()
    # Approximate edge length to hit target vertex count
    target_len = diag / (target_vertices ** 0.5)

    ms.meshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.PureValue(target_len),
        iterations=5
    )

    result_vertices = ms.current_mesh().vertex_number()
    print(f"Result vertices: {result_vertices}")

    ms.save_current_mesh(output_obj)

    print(f"Converting {output_obj} -> {output_glb}")
    obj_to_glb(output_obj, output_glb)
    print("Done.")

    os.remove(input_obj)
    os.remove(output_obj)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    input_glb = sys.argv[1]
    output_glb = sys.argv[2] if len(sys.argv) > 2 else input_glb.replace(".glb", "_remeshed.glb")
    remesh(input_glb, output_glb)
