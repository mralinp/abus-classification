import trimesh
import numpy as np
import skimage

def get_mesh_from_3d_mask(binary_mask_3d: np.ndarray) -> trimesh.Trimesh:
    verts, faces, _, _ = skimage.measure.marching_cubes(binary_mask_3d, level=0.5)
    # Create a Trimesh object from vertices and faces
    tumor_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # Visualize the tumor mesh
    return tumor_mesh