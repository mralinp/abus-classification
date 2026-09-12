import trimesh
import numpy as np
from skimage import measure, morphology
from typing import Optional, Tuple

def preprocess_binary_mask(mask: np.ndarray, 
                         smooth: bool = True,
                         remove_small_objects: bool = True,
                         min_size: int = 64) -> np.ndarray:
    """
    Preprocess a binary mask to improve mesh quality
    
    Parameters:
        mask: Binary mask (0s and 1s)
        smooth: Whether to apply gaussian smoothing
        remove_small_objects: Whether to remove small disconnected components
        min_size: Minimum size of objects to keep
    
    Returns:
        Preprocessed binary mask
    """
    if not mask.dtype == bool:
        mask = mask.astype(bool)
    
    if remove_small_objects:
        # scikit-image >=0.26 renamed min_size to max_size and made the bound
        # inclusive, so max_size=min_size-1 keeps the original semantics.
        mask = morphology.remove_small_objects(mask, max_size=min_size - 1)
    
    if smooth:
        # Apply binary closing to smooth boundaries
        mask = morphology.closing(mask)
    
    return mask

def get_mesh_from_3d_mask(binary_mask_3d: np.ndarray,
                         spacing: Optional[Tuple[float, float, float]] = None,
                         level: float = 0.5,
                         preprocess: bool = True) -> trimesh.Trimesh:
    """
    Convert a 3D binary mask to a triangle mesh using marching cubes
    
    Parameters:
        binary_mask_3d: 3D numpy array with binary values (0 and 1)
        spacing: Voxel spacing in each dimension (x, y, z). If None, assumes isotropic spacing of (1,1,1)
        level: Threshold level for marching cubes
        preprocess: Whether to preprocess the mask for better mesh quality
    
    Returns:
        trimesh.Trimesh object representing the surface
    
    Raises:
        ValueError: If input mask is invalid or empty after preprocessing
    """
    if binary_mask_3d.ndim != 3:
        raise ValueError(f"Expected 3D array, got {binary_mask_3d.ndim}D array")
    
    if not np.any(binary_mask_3d):
        raise ValueError("Input mask is empty (all zeros)")
    
    if preprocess:
        binary_mask_3d = preprocess_binary_mask(binary_mask_3d)
        
        if not np.any(binary_mask_3d):
            raise ValueError("Mask is empty after preprocessing")
    
    # Set default spacing if not provided
    if spacing is None:
        spacing = (1.0, 1.0, 1.0)
    
    try:
        # Generate surface mesh using marching cubes
        verts, faces, normals, _ = measure.marching_cubes(binary_mask_3d, level=level, spacing=spacing)
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=verts, 
                             faces=faces,
                             vertex_normals=normals)
        
        # Basic mesh cleanup (trimesh >=4.0 replaced the remove_* helpers
        # with update_faces() over an explicit face mask)
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        
        # Ensure consistent face winding and normals
        mesh.fix_normals()
        
        return mesh
        
    except Exception as e:
        raise ValueError(f"Failed to generate mesh: {str(e)}")

def lesion_mesh(mask: np.ndarray,
                spacing: Optional[Tuple[float, float, float]] = None,
                pad: int = 2) -> trimesh.Trimesh:
    """
    Build a watertight surface mesh for a lesion mask.

    This is `get_mesh_from_3d_mask` with the two settings lesion masks need:

    - The mask is zero-padded by `pad` voxels. Masks cropped to their bounding
      box touch the array border, and marching cubes leaves the surface open
      there. The mesh is translated back afterwards, so vertex coordinates stay
      in the original mask's frame.
    - Preprocessing is off. Its morphological closing pinches thin structures
      into non-manifold geometry; on the TDSC-ABUS masks it leaves 8 of 200
      surfaces open, with inverted winding and a negative volume.

    Parameters:
        mask: 3D binary lesion mask
        spacing: Voxel spacing in each dimension. Defaults to (1, 1, 1).
        pad: Zero voxels added on every side before meshing

    Returns:
        trimesh.Trimesh object representing the closed lesion surface
    """
    spacing = (1.0, 1.0, 1.0) if spacing is None else tuple(spacing)
    padded = np.pad((np.asarray(mask) > 0).astype(np.uint8), pad)
    mesh = get_mesh_from_3d_mask(padded, spacing=spacing, level=0.5, preprocess=False)
    mesh.apply_translation(-pad * np.asarray(spacing, dtype=float))
    return mesh
