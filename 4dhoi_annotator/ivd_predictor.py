"""
IVD (InterActVLM-Discrete) Model Predictor for 4DHOI Annotation Tool

This module provides a wrapper for the InterPoint model to predict
human contact joints and corresponding 3D coordinates on object surfaces.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add InterPoint model to path (sibling directory in the repo)
INTERPOINT_PATH = Path(__file__).resolve().parent.parent / "interpoint"
if not INTERPOINT_PATH.exists():
    # Fallback: check environment variable
    _env_path = os.environ.get("INTERPOINT_DIR")
    if _env_path:
        INTERPOINT_PATH = Path(_env_path)

if str(INTERPOINT_PATH) not in sys.path:
    sys.path.insert(0, str(INTERPOINT_PATH))

import torch
from PIL import Image


class IVDPredictor:
    """
    Wrapper for InterPoint model inference.

    Provides lazy loading of the model and prediction API for
    human contact joints and object surface coordinates.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda:0',
        use_lightweight_vlm: bool = False,
        d_tr: int = 256,
        num_body_points: int = 87,
        num_object_queries: int = 87,
    ):
        """
        Initialize the predictor.

        Args:
            checkpoint_path: Path to model checkpoint. If None, uses default.
            device: Device to run inference on ('cuda:0' or 'cpu').
            use_lightweight_vlm: Whether to use lightweight VLM (faster).
            d_tr: Transformer dimension.
            num_body_points: Number of body keypoints (87 for SMPL-X parts).
            num_object_queries: Number of object queries.
        """
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        self.checkpoint_path = checkpoint_path or self._get_default_checkpoint()
        self.use_lightweight_vlm = use_lightweight_vlm
        self.d_tr = d_tr
        self.num_body_points = num_body_points
        self.num_object_queries = num_object_queries

        self.model = None
        self.joint_names = None
        self._loaded = False

    def _get_default_checkpoint(self) -> str:
        """Get the default checkpoint path."""
        default_candidates = [
            INTERPOINT_PATH / "checkpoints" / "epoch_078.pth",
            INTERPOINT_PATH / "checkpoints" / "4dhoi_contrastive_from_scratch" / "epoch_078.pth",
        ]
        for default_path in default_candidates:
            if default_path.exists():
                return str(default_path)
        # Fallback options
        fallback_paths = [
            INTERPOINT_PATH / "checkpoints" / "4dhoi_only" / "epoch_80.pth",
            INTERPOINT_PATH / "checkpoints" / "intrcap_modify" / "epoch_060.pth",
        ]
        for fallback_path in fallback_paths:
            if fallback_path.exists():
                return str(fallback_path)
        raise FileNotFoundError(
            f"No checkpoint found. Please specify checkpoint_path. "
            f"Checked: {default_candidates + fallback_paths}"
        )

    def _load_joint_names(self) -> List[str]:
        """Load joint names from part_kp.json."""
        part_kp_path = INTERPOINT_PATH / "data" / "part_kp.json"
        if not part_kp_path.exists():
            # Fallback: try local solver data
            part_kp_path = Path(__file__).resolve().parent / "solver" / "data" / "part_kp.json"
        if not part_kp_path.exists():
            raise FileNotFoundError(f"part_kp.json not found at {part_kp_path}")

        with open(part_kp_path, 'r') as f:
            part_kp = json.load(f)

        joint_list = list(part_kp.keys())
        return joint_list

    def _load_model(self):
        """Lazy load the model on first use."""
        if self._loaded:
            return

        print(f"[IVDPredictor] Loading model from {self.checkpoint_path}...")

        from models import build_model

        config = {
            'd_tr': self.d_tr,
            'num_body_points': self.num_body_points,
            'num_object_queries': self.num_object_queries,
            'use_lightweight_vlm': self.use_lightweight_vlm,
            'freeze_vlm': True,
            'device': self.device,
            'vlm_device_map': self.device,
        }

        self.model = build_model(config)

        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False
            )
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=False)
            print(f"[IVDPredictor] Loaded checkpoint from {self.checkpoint_path}")
        else:
            print(f"[IVDPredictor] Warning: Checkpoint not found at {self.checkpoint_path}")

        self.model = self.model.to(self.device)
        self.model.eval()

        self.joint_names = self._load_joint_names()
        print(f"[IVDPredictor] Loaded {len(self.joint_names)} joint names")

        self._loaded = True
        print(f"[IVDPredictor] Model loaded successfully on {self.device}")

    def preprocess_image(self, rgb_frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess RGB image for model input.

        Args:
            rgb_frame: RGB image as numpy array (H, W, 3), values 0-255.

        Returns:
            Preprocessed image tensor (1, 3, 224, 224).
        """
        if rgb_frame.dtype != np.uint8:
            rgb_frame = (rgb_frame * 255).astype(np.uint8)

        img = Image.fromarray(rgb_frame)
        img = img.resize((224, 224))

        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

        return img_tensor.to(self.device)

    def preprocess_points(self, vertices: np.ndarray, num_points: int = 1024) -> torch.Tensor:
        """
        Preprocess object vertices for model input.

        Args:
            vertices: Object mesh vertices as numpy array (N, 3).
            num_points: Number of points to sample.

        Returns:
            Point cloud tensor (1, num_points, 3).
        """
        vertices = np.asarray(vertices, dtype=np.float32)

        if len(vertices) == 0:
            return torch.zeros((1, num_points, 3), dtype=torch.float32, device=self.device)

        center = np.mean(vertices, axis=0)
        vertices_centered = vertices - center

        if len(vertices_centered) >= num_points:
            indices = np.random.choice(len(vertices_centered), num_points, replace=False)
        else:
            indices = np.random.choice(len(vertices_centered), num_points, replace=True)

        points = vertices_centered[indices]

        points_tensor = torch.from_numpy(points).unsqueeze(0).float()

        return points_tensor.to(self.device)

    def predict(
        self,
        rgb_frame: np.ndarray,
        object_vertices: np.ndarray,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Run prediction on a single frame.

        Args:
            rgb_frame: RGB image as numpy array (H, W, 3).
            object_vertices: Object mesh vertices as numpy array (N, 3).
            threshold: Confidence threshold for filtering predictions.
            top_k: If specified, return only top K predictions by confidence.

        Returns:
            List of prediction dicts with keys: joint, xyz, confidence, joint_idx.
        """
        self._load_model()

        rgb_tensor = self.preprocess_image(rgb_frame)
        points_tensor = self.preprocess_points(object_vertices)

        with torch.no_grad():
            outputs = self.model(rgb_tensor, points_tensor, return_aux=False)

        human_contact = outputs['human_contact'].cpu().numpy()[0]
        object_coords = outputs['object_coords'].cpu().numpy()[0]

        predictions = []
        for i, (prob, coords) in enumerate(zip(human_contact, object_coords)):
            if prob >= threshold:
                joint_name = self.joint_names[i] if i < len(self.joint_names) else f"joint_{i}"
                predictions.append({
                    'joint': joint_name,
                    'xyz': coords.tolist(),
                    'confidence': float(prob),
                    'joint_idx': i,
                })

        predictions.sort(key=lambda x: x['confidence'], reverse=True)

        if top_k is not None and len(predictions) > top_k:
            predictions = predictions[:top_k]

        return predictions

    def transform_coords_to_mesh(
        self,
        predictions: List[Dict],
        mesh_vertices: np.ndarray,
        mesh_center: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Transform predicted coordinates back to mesh coordinate space.

        Args:
            predictions: List of prediction dicts from predict().
            mesh_vertices: Original mesh vertices (N, 3).
            mesh_center: Optional center point. If None, computed from vertices.

        Returns:
            Updated predictions with xyz in mesh coordinate space.
        """
        mesh_vertices = np.asarray(mesh_vertices)

        if mesh_center is None:
            mesh_center = np.mean(mesh_vertices, axis=0)

        updated = []
        for pred in predictions:
            new_pred = pred.copy()
            xyz = np.array(pred['xyz'])
            new_pred['xyz'] = (xyz + mesh_center).tolist()
            updated.append(new_pred)

        return updated

    def find_nearest_vertex(
        self,
        xyz: List[float],
        mesh_vertices: np.ndarray,
    ) -> Tuple[int, float]:
        """
        Find the nearest vertex index in the mesh to a given 3D point.

        Args:
            xyz: Target point [x, y, z].
            mesh_vertices: Mesh vertices (N, 3).

        Returns:
            Tuple of (vertex_index, distance).
        """
        mesh_vertices = np.asarray(mesh_vertices)
        xyz = np.array(xyz)

        distances = np.linalg.norm(mesh_vertices - xyz, axis=1)
        min_idx = np.argmin(distances)

        return int(min_idx), float(distances[min_idx])

    def get_joint_names(self) -> List[str]:
        """Get list of all joint names."""
        self._load_model()
        return self.joint_names.copy()


if __name__ == "__main__":
    print("Testing IVDPredictor...")

    predictor = IVDPredictor(use_lightweight_vlm=True)

    dummy_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_vertices = np.random.randn(1000, 3).astype(np.float32)

    predictions = predictor.predict(dummy_rgb, dummy_vertices, threshold=0.3)

    print(f"Got {len(predictions)} predictions:")
    for pred in predictions[:5]:
        print(f"  - {pred['joint']}: {pred['confidence']:.3f} at {pred['xyz']}")
