import copy
import os
import sys

import numpy as np
import open3d as o3d
import smplx
import torch


def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # type: ignore
    except Exception:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    primary = os.path.join(base_path, relative_path)
    if os.path.exists(primary):
        return primary

    # Fallback for missing local SMPL-X model file.
    if relative_path.replace('\\', '/').endswith('video_optimizer/smpl_models/SMPLX_NEUTRAL.npz'):
        env_model = os.environ.get("SMPLX_MODEL")
        if env_model and os.path.exists(env_model):
            return env_model
        repo_root = os.path.dirname(base_path)
        shared_model = os.path.join(repo_root, "shared_data", "SMPLX_NEUTRAL.npz")
        if os.path.exists(shared_model):
            return shared_model

    return primary


def _resolve_smplx_model_path() -> str:
    """Resolve SMPL-X model path with robust fallbacks."""
    candidates = []

    env_path = os.environ.get("SMPLX_MODEL")
    if env_path:
        candidates.append(env_path)

    # Original local path used by this module
    candidates.append(resource_path("video_optimizer/smpl_models/SMPLX_NEUTRAL.npz"))

    # Repository-level shared_data fallback
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates.append(os.path.join(repo_root, "shared_data", "SMPLX_NEUTRAL.npz"))

    for p in candidates:
        if p and os.path.exists(p):
            return p

    # Keep original path in error message if nothing found
    return candidates[1]


_model_type = "smplx"
_model_folder = _resolve_smplx_model_path()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smplx.create(
    _model_folder,
    model_type=_model_type,
    gender="neutral",
    num_betas=10,
    num_expression_coeffs=10,
    use_pca=False,
    flat_hand_mean=True,
).to(_device)


def apply_initial_transform_to_mesh(mesh: o3d.geometry.TriangleMesh, t: np.ndarray):
    mesh_copy = copy.deepcopy(mesh)
    verts = np.asarray(mesh_copy.vertices)
    transformed_verts = verts + t
    mesh_copy.vertices = o3d.utility.Vector3dVector(transformed_verts)
    return mesh_copy


def apply_initial_transform_to_points(points: np.ndarray, t: np.ndarray) -> np.ndarray:
    return points + t
