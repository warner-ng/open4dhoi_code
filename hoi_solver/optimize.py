# -*- coding: utf-8 -*-
"""
HOI Optimization Pipeline

Supports:
- Single video mode: --data_dir <session_folder>
- Batch mode: --batch (process all records with annotation_progress==3.0)
- Daemon mode: --daemon (continuously poll for new tasks)

Automatically converts annotations from decimated mesh to original mesh if needed.
"""

import argparse
import fcntl
import json
import os
import signal
import sys
import time
import cv2
import numpy as np
import torch
import open3d as o3d
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from video_optimizer.kp_use_new import kp_use_new
from video_optimizer.kp_common import resource_path
from video_optimizer.utils.dataset_util import (
    get_static_flag_from_merged,
    validate_dof_from_merged,
)

# ============= Path Configuration =============

PROJECT_DIR = Path(__file__).resolve().parent
# Default: upload_records.json is in 4dhoi_autorecon (sibling directory)
DEFAULT_UPLOAD_RECORDS = PROJECT_DIR.parent / "4dhoi_autorecon" / "upload_records.json"

# Task name for status reporting
TASK_NAME = "hoi_optimize"

# ============= File Locking Utilities =============

@contextmanager
def _file_lock(path: Path, exclusive: bool = True):
    """File-level lock for concurrent safety"""
    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as f:
        lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(f.fileno(), lock_type)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_upload_records(records_path: Union[str, Path]) -> List[dict]:
    """Load upload_records.json with file lock"""
    records_path = Path(records_path)
    with _file_lock(records_path, exclusive=False):
        if not records_path.exists():
            return []
        try:
            with records_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError):
            return []


def _write_records_unsafe(records: List[dict], records_path: Path) -> None:
    """Write upload_records.json (internal, no lock)"""
    tmp_path = records_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)
    tmp_path.replace(records_path)


def atomic_update_progress(records_path: Union[str, Path], session_folder: str, new_progress: float) -> None:
    """Atomically update annotation_progress for a single record"""
    records_path = Path(records_path)
    with _file_lock(records_path, exclusive=True):
        if not records_path.exists():
            return
        try:
            with records_path.open("r", encoding="utf-8") as f:
                records = json.load(f)
            if not isinstance(records, list):
                return
        except (json.JSONDecodeError, IOError):
            return

        updated = False
        for rec in records:
            if str(rec.get("session_folder", "")) == session_folder:
                rec["annotation_progress"] = new_progress
                updated = True
                break

        if updated:
            _write_records_unsafe(records, records_path)


# ============= Task Status Reporting =============

# Try to import TaskProgress from 4dhoi_autorecon for status reporting
try:
    sys.path.insert(0, str(PROJECT_DIR.parent / "4dhoi_autorecon"))
    from task_status import TaskProgress, update_status, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED
    TASK_STATUS_AVAILABLE = True
except ImportError:
    TASK_STATUS_AVAILABLE = False

    class TaskProgress:
        """Dummy TaskProgress when task_status module is not available"""
        def __init__(self, task_name: str):
            self.task_name = task_name
        def start(self, total: int = 0, message: str = ""):
            print(f"[{self.task_name}] Start: {message}")
        def update(self, current: int, item_name: str = "", message: str = ""):
            print(f"[{self.task_name}] Progress {current}: {item_name}")
        def complete(self, message: str = ""):
            print(f"[{self.task_name}] Complete: {message}")
        def fail(self, error: str):
            print(f"[{self.task_name}] Failed: {error}")


# ============= Convert Annotations (decimated mesh → original mesh) =============

def _load_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_annotations(video_dir: str) -> bool:
    """
    Convert annotations from decimated mesh vertex indices to original mesh indices.

    The annotation tool works on a decimated mesh for performance. This function
    maps those indices back to the original high-res mesh using nearest-neighbor
    search, producing kp_record_new.json from kp_record_merged.json.

    Returns True on success, False on failure.
    """
    video_dir = Path(video_dir)
    obj_org_path = video_dir / "obj_org.obj"
    kp_merged_path = video_dir / "kp_record_merged.json"
    kp_new_path = video_dir / "kp_record_new.json"

    if not obj_org_path.exists():
        print(f"[convert] obj_org.obj not found: {obj_org_path}")
        return False
    if not kp_merged_path.exists():
        print(f"[convert] kp_record_merged.json not found: {kp_merged_path}")
        return False

    try:
        # Load original mesh
        mesh_org = o3d.io.read_triangle_mesh(str(obj_org_path))
        print(f"[convert] Original mesh: {len(mesh_org.vertices)} vertices")

        # Decimate to match annotator settings
        target_faces = 60000
        vert_threshold = 30000
        mesh_decimated = o3d.geometry.TriangleMesh(mesh_org)
        if len(mesh_org.vertices) > vert_threshold:
            mesh_decimated = mesh_decimated.simplify_quadric_decimation(
                target_number_of_triangles=target_faces
            )
        print(f"[convert] Decimated mesh: {len(mesh_decimated.vertices)} vertices")

        # Build vertex mapping: decimated index → original index (nearest neighbor)
        org_tree = o3d.geometry.KDTreeFlann(mesh_org)
        decimated_verts = np.asarray(mesh_decimated.vertices)
        mapping = {}
        for dec_idx in range(len(decimated_verts)):
            result = org_tree.search_knn_vector_3d(decimated_verts[dec_idx], 1)
            mapping[dec_idx] = result[1][0]
        print(f"[convert] Built mapping: {len(mapping)} decimated → original vertices")

        # Load and convert annotations
        kp_data = _load_json(kp_merged_path)
        kp_new = {}
        conversion_count = 0

        for frame_key, frame_data in kp_data.items():
            # Copy metadata fields as-is
            if frame_key in ('object_scale', 'is_static_object', 'start_frame_index'):
                kp_new[frame_key] = frame_data
                continue
            if not isinstance(frame_data, dict):
                kp_new[frame_key] = frame_data
                continue

            frame_new = {}

            # Convert 2D keypoint indices
            two_d = frame_data.get('2D_keypoint', [])
            if two_d and isinstance(two_d, list):
                two_d_new = []
                for item in two_d:
                    if isinstance(item, list) and len(item) >= 1:
                        dec_idx = item[0]
                        if isinstance(dec_idx, int) and dec_idx in mapping:
                            two_d_new.append([mapping[dec_idx]] + item[1:])
                            conversion_count += 1
                        else:
                            two_d_new.append(item)
                    else:
                        two_d_new.append(item)
                if two_d_new:
                    frame_new['2D_keypoint'] = two_d_new

            # Convert 3D joint indices
            for key, value in frame_data.items():
                if key == '2D_keypoint':
                    continue
                if isinstance(value, int) and value in mapping:
                    frame_new[key] = mapping[value]
                    conversion_count += 1
                else:
                    frame_new[key] = value

            kp_new[frame_key] = frame_new

        # Save converted annotations
        with open(kp_new_path, 'w', encoding='utf-8') as f:
            json.dump(kp_new, f, indent=2, ensure_ascii=False)

        print(f"[convert] Converted {conversion_count} indices, saved: {kp_new_path}")
        return True

    except Exception as e:
        print(f"[convert] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def ensure_kp_record_new(video_dir: str) -> bool:
    """
    Ensure kp_record_new.json exists in video_dir.
    If missing, convert from kp_record_merged.json by mapping decimated
    mesh indices to original mesh indices.

    Returns True if kp_record_new.json exists (or was created), False otherwise.
    """
    video_dir_p = Path(video_dir)
    kp_new_path = video_dir_p / "kp_record_new.json"

    if kp_new_path.exists():
        print(f"[INFO] kp_record_new.json already exists, skipping conversion.")
        return True

    kp_merged_path = video_dir_p / "kp_record_merged.json"
    if not kp_merged_path.exists():
        print(f"[ERROR] Neither kp_record_new.json nor kp_record_merged.json found in {video_dir}")
        return False

    print(f"[INFO] kp_record_new.json not found, converting from kp_record_merged.json...")
    return convert_annotations(video_dir)


# ============= Mesh Preprocessing (matching annotation app) =============

def preprocess_obj_for_optimization(obj_org, obj_poses_data, merged_data):
    """
    Apply the same preprocessing as annotation app to match vertex coordinates.

    This applies:
    1. Rotation (if present in obj_poses)
    2. Scale from obj_poses
    3. Recenter by mean
    4. Apply object_scale from merged data
    5. Translate by t from obj_poses

    Args:
        obj_org: Original mesh from obj_org.obj
        obj_poses_data: Data from obj_poses.json
        merged_data: Data from kp_record_new.json (contains object_scale)

    Returns:
        Preprocessed mesh with vertices matching annotation app
    """
    mesh = deepcopy(obj_org)

    # 1. Apply rotation if present (from obj_poses)
    # Note: In annotation app, rotation is per-frame, but for static objects we use first frame
    # For simplicity, we skip rotation here as it's usually not used

    # 2. Apply scale from obj_poses
    scale = float(obj_poses_data.get('scale', 1.0))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    vertices *= scale

    # 3. Recenter by mean (same as annotation app)
    center = np.mean(vertices, axis=0)
    vertices = vertices - center

    # 4. Apply object_scale from merged data
    object_scale = float(merged_data.get('object_scale', 1.0))
    vertices *= object_scale

    # 5. Translate by t from obj_poses (initial position from depth alignment)
    t_raw = obj_poses_data.get('t', None)
    if t_raw is not None:
        if isinstance(t_raw, list) and t_raw and isinstance(t_raw[0], list):
            t0 = np.array(t_raw[0], dtype=np.float32)
        elif isinstance(t_raw, list) and len(t_raw) == 3:
            t0 = np.array(t_raw, dtype=np.float32)
        else:
            t0 = np.zeros(3, dtype=np.float32)
        vertices = vertices + t0

    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))

    print(f"[INFO] Preprocessed mesh: scale={scale}, object_scale={object_scale}, t={t_raw}")
    return mesh


# ============= Video Assembly =============

def assemble_preview_video(optimized_dir: str, output_path: str, fps: float = 18.0) -> None:
    frame_files = sorted(
        [os.path.join(optimized_dir, f) for f in os.listdir(optimized_dir) if f.endswith(".png")]
    )
    if not frame_files:
        print("No frames found for video")
        return
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print("Cannot read first frame, cancel video")
        return
    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for ff in frame_files:
        fr = cv2.imread(ff)
        if fr is not None:
            out.write(fr)
    out.release()
    print(f"Preview video saved to: {output_path}")


# ============= Core Optimization =============

def optimize_single_record(
    record,
    max_frames: Optional[int] = None,
    save_ls_meshes: bool = False,
    ls_mesh_dir: str = "debug_ls_meshes",
    start_frame_override: Optional[int] = None,
    end_frame_exclusive_override: Optional[int] = None,
    use_least_squares_only: bool = False,
    best_frame: Optional[int] = None,
):
    video_dir = record
    print(f"Video directory: {video_dir}")
    video_path = Path(video_dir, "video.mp4")
    if not video_path.exists():
        print(f"Cannot find video: {video_path}")
        return False
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    if total_frames <= 0:
        print("Cannot read video or video is empty")
        return False

    # Ensure kp_record_new.json exists (auto-convert from kp_record_merged.json if needed)
    if not ensure_kp_record_new(video_dir):
        print("Cannot create kp_record_new.json, terminate optimization")
        return False
    merged_path = os.path.join(video_dir, "kp_record_new.json")
    with open(merged_path, "r", encoding="utf-8") as f:
        merged = json.load(f)

    static_object = get_static_flag_from_merged(merged_path)
    print(f"Static object: {static_object}")
    try:
        keys_int = sorted(int(k) for k in merged.keys() if str(k).isdigit())
    except Exception:
        keys_int = []
    if not keys_int:
        print("Merged annotation is empty, terminate.")
        return False

    # Use the first annotated frame as start (not start_frame_index which may be stale)
    start_frame_default = min(keys_int)

    # Use end_frame_exclusive consistently across the pipeline.
    end_frame_exclusive_default = max(keys_int) + 1

    start_frame = start_frame_default
    end_frame_exclusive = end_frame_exclusive_default
    if start_frame_override is not None:
        start_frame = int(start_frame_override)
    if end_frame_exclusive_override is not None:
        end_frame_exclusive = int(end_frame_exclusive_override)

    # Clamp to available annotated range
    start_frame = max(start_frame, start_frame_default)
    end_frame_exclusive = min(end_frame_exclusive, end_frame_exclusive_default)
    if end_frame_exclusive <= start_frame:
        print(f"Invalid frame range: start={start_frame}, end_exclusive={end_frame_exclusive}")
        return False
    if max_frames is not None and max_frames > 0:
        end_frame_exclusive = min(end_frame_exclusive, start_frame + int(max_frames))
    end_frame_inclusive = max(start_frame, end_frame_exclusive - 1)

    print(f"Optimized frame range: {start_frame}..{end_frame_inclusive} (total={end_frame_exclusive - start_frame})")
    if not validate_dof_from_merged(merged, start_frame, end_frame_inclusive):
        print("Invalid annotation frames, terminate optimization")
        return False

    # Prefer result_hand.pt (contains hand params from make_hand_sam3d); no update_hand_pose.
    motion_dir = os.path.join(video_dir, "motion")
    result_hand_path = os.path.join(motion_dir, "result_hand.pt")
    result_pt_path = os.path.join(motion_dir, "result.pt")
    if os.path.exists(result_hand_path):
        output = torch.load(result_hand_path, map_location="cpu", weights_only=False)
        print("[INFO] Using result_hand.pt (hand params included)")
    else:
        output = torch.load(result_pt_path, map_location="cpu", weights_only=False)
        print("[INFO] Using result.pt (no result_hand.pt found)")

    body_params = output["smpl_params_incam"]
    global_body_params = output["smpl_params_global"]
    K = output["K_fullimg"][0]
    t_len = body_params["body_pose"].shape[0]
    with open(resource_path("video_optimizer/data/part_kp.json"), "r", encoding="utf-8") as f:
        human_part = json.load(f)

    # Build hand_poses directly from result (result_hand.pt has left/right_hand_pose); do not use update_hand_pose.
    hand_poses = {}
    incam = output.get("smpl_params_incam", body_params)
    left_hand = incam.get("left_hand_pose")
    right_hand = incam.get("right_hand_pose")
    if left_hand is not None and right_hand is not None and left_hand.shape[0] == t_len and right_hand.shape[0] == t_len:
        for i in range(t_len):
            hand_poses[str(i)] = {
                "left_hand": left_hand[i].cpu().numpy().tolist() if torch.is_tensor(left_hand[i]) else left_hand[i],
                "right_hand": right_hand[i].cpu().numpy().tolist() if torch.is_tensor(right_hand[i]) else right_hand[i],
            }
        print(f"[INFO] Hand poses from result: {len(hand_poses)} frames")
    else:
        # Fallback: zero hand pose (e.g. old result.pt without hand keys)
        for i in range(min(total_frames, end_frame_exclusive)):
            hand_poses[str(i)] = {
                "left_hand": [[0.0, 0.0, 0.0]] * 15,
                "right_hand": [[0.0, 0.0, 0.0]] * 15,
            }
        print(f"[INFO] Hand poses: zero fallback for {len(hand_poses)} frames (no hand params in result)")

    # Use obj_org.obj because kp_record_new.json contains indices on the original mesh
    obj_org_path = os.path.join(video_dir, "obj_org.obj")
    if not os.path.exists(obj_org_path):
        print(f"Cannot find obj_org.obj: {obj_org_path}")
        return False

    obj_org_raw = o3d.io.read_triangle_mesh(obj_org_path)
    print(f"[INFO] Loaded obj_org.obj with {len(obj_org_raw.vertices)} vertices")

    # Load obj_poses.json to get scale and other parameters
    obj_poses_path = os.path.join(video_dir, "align", "obj_poses.json")
    if not os.path.exists(obj_poses_path):
        # Try output directory as fallback
        obj_poses_path = os.path.join(video_dir, "output", "obj_poses.json")

    if os.path.exists(obj_poses_path):
        with open(obj_poses_path, "r", encoding="utf-8") as f:
            obj_poses_data = json.load(f)
    else:
        print(f"[WARN] obj_poses.json not found, using default scale=1.0")
        obj_poses_data = {"scale": 1.0}

    # Preprocess mesh to match annotation app (apply scale, recenter, apply object_scale)
    obj_org = preprocess_obj_for_optimization(obj_org_raw, obj_poses_data, merged)
    sampled_obj = obj_org.simplify_quadric_decimation(target_number_of_triangles=1000)

    seq_len = end_frame_exclusive - start_frame
    obj_orgs = [deepcopy(obj_org) for _ in range(max(seq_len, 1))]
    sampled_orgs = [deepcopy(sampled_obj) for _ in range(max(seq_len, 1))]
    body_params_new, hand_poses_new, Rf_seg, tf_seg, optimized_params = kp_use_new(
        output=output,
        hand_poses=hand_poses,
        body_poses=body_params,
        global_body_poses=global_body_params,
        obj_orgs=obj_orgs,
        sampled_orgs=sampled_orgs,
        human_part=human_part,
        K=K,
        start_frame=start_frame,
        end_frame=end_frame_exclusive,
        video_dir=video_dir,
        is_static_object=static_object,
        kp_record_path=merged_path,
        save_ls_meshes=save_ls_meshes,
        ls_mesh_dir=os.path.join(video_dir, ls_mesh_dir),
        use_least_squares_only=use_least_squares_only,
        best_frame_override=best_frame
    )
    print("Optimize finished")
    optimized_dir = os.path.join(video_dir, "optimized_frames")
    preview_path = os.path.join(video_dir, "optimized_preview.mp4")
    # Only generate preview video if optimized_frames directory exists
    if os.path.exists(optimized_dir) and os.path.isdir(optimized_dir):
        assemble_preview_video(optimized_dir, preview_path, fps=18.0)
    else:
        print(f"[INFO] Skipping preview video generation (optimized_frames not found, likely due to use_least_squares_only=True)")
    save_dir = os.path.join(video_dir, "final_optimized_parameters")
    os.makedirs(save_dir, exist_ok=True)

    def _to_list(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy().tolist()
        if hasattr(x, "tolist"):
            return x.tolist()
        return x

    final_human_params = {
        "body_pose": {},
        "betas": {},
        "global_orient": {},
        "transl": {},
        "left_hand_pose": {},
        "right_hand_pose": {},
    }
    hp = hand_poses_new
    print('check', len(body_params_new["body_pose"]), len(hp), start_frame, end_frame_exclusive)
    for fi in range(start_frame, end_frame_exclusive):
        idx = fi
        print('save', idx)
        final_human_params["body_pose"][str(fi)] = _to_list(body_params_new["body_pose"][fi-start_frame])
        final_human_params["betas"][str(fi)] = _to_list(body_params_new["betas"][fi-start_frame])
        final_human_params["global_orient"][str(fi)] = _to_list(body_params_new["global_orient"][fi-start_frame])
        final_human_params["transl"][str(fi)] = _to_list(body_params_new["transl"][fi-start_frame])
        final_human_params["left_hand_pose"][str(fi)] = _to_list(hp[fi-start_frame]["left_hand"])
        final_human_params["right_hand_pose"][str(fi)] = _to_list(hp[fi-start_frame]["right_hand"])
    final_object_params = {
        "poses": {},
        "centers": {},
    }
    for local_i, fi in enumerate(range(start_frame, end_frame_exclusive)):
        final_object_params["poses"][str(fi)] = np.asarray(Rf_seg[local_i]).tolist()
        final_object_params["centers"][str(fi)] = np.asarray(tf_seg[local_i]).tolist()
    incam_subset = {
        "global_orient": [],
        "transl": []
    }
    global_subset = {
        "global_orient": [],
        "transl": []
    }
    for fi in range(start_frame, end_frame_exclusive):
        incam_subset["global_orient"].append(_to_list(output["smpl_params_incam"]["global_orient"][fi]))
        incam_subset["transl"].append(_to_list(output["smpl_params_incam"]["transl"][fi]))
        global_subset["global_orient"].append(_to_list(output["smpl_params_global"]["global_orient"][fi]))
        global_subset["transl"].append(_to_list(output["smpl_params_global"]["transl"][fi]))

    original_object_path = os.path.join(video_dir, "obj_org.obj")
    if not os.path.isfile(original_object_path):
        raise FileNotFoundError(f"Object mesh not found: {original_object_path}")
    saved_object_path = original_object_path
    transformed_summary = {
        "object_mesh_path": saved_object_path,
        "note": "No transformed_parameters saved. Render should compute R_total/T_total from raw object params + camera subsets for consistency with preview."
    }
    kp_records_merged = {f"{fi:05d}": merged.get(f"{fi:05d}", {"2D_keypoint": []}) for fi in range(start_frame, end_frame_exclusive)}
    combined_payload = {
        "metadata": {
            "description": "All parameters in one file: raw params, incam/global camera subsets, kp_records, and mesh path. Render recomputes object transforms for consistency.",
            "frame_range": {"start": start_frame, "end": end_frame_exclusive},
            "save_dir": save_dir
        },
        "camera": {
            "K": _to_list(K),
            "note": "Intrinsic matrix used during optimization (K_fullimg[0] from motion/result.pt)."
        },
        "human_params_raw": final_human_params,
        "object_params_raw": final_object_params,
        "smpl_params_incam_subset": incam_subset,
        "smpl_params_global_subset": global_subset,
        "kp_records": kp_records_merged,
        "transformed": transformed_summary
    }
    combined_path = os.path.join(save_dir, "all_parameters_latest.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_payload, f, indent=2, ensure_ascii=False)
    print(f"Saved parameters for rendering: {combined_path}")
    summary = {
        "frame_range": {"start": start_frame, "end": end_frame_exclusive},
        "static_object": static_object,
        "optimized_params_available": bool(optimized_params),
    }
    with open(os.path.join(video_dir, "optimize_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return True


# ============= Daemon Mode =============

def daemon_loop(
    records_path: Path,
    poll_interval: int = 120,
    args: Any = None,
):
    """
    Daemon loop: continuously poll for records with annotation_progress==3.0
    and process them, updating progress to 4.0 on success.
    """
    # Signal handling for graceful exit
    stop_requested = [False]

    def signal_handler(signum, frame):
        signame = signal.Signals(signum).name
        print(f"\n[{TASK_NAME}] Received {signame}, will exit after current task...")
        stop_requested[0] = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"[{TASK_NAME}] Daemon started")
    print(f"[{TASK_NAME}] Monitoring: annotation_progress 3.0 -> 4.0")
    print(f"[{TASK_NAME}] Poll interval: {poll_interval} seconds")
    print(f"[{TASK_NAME}] Records file: {records_path}")
    print(f"[{TASK_NAME}] Press Ctrl+C to exit gracefully")
    print()

    # Initialize progress reporter
    progress_reporter = TaskProgress(TASK_NAME)

    # Adaptive logging: reduce log frequency when no tasks
    empty_poll_count = 0

    # Extract optimization args
    start_override = getattr(args, 'start_frame', None)
    end_exclusive_override = getattr(args, 'end_frame_exclusive', None)
    max_frames = getattr(args, 'max_frames', None)
    save_ls_meshes = getattr(args, 'save_ls_meshes', False)
    ls_mesh_dir = getattr(args, 'ls_mesh_dir', 'debug_ls_meshes')
    use_least_squares_only = getattr(args, 'use_least_squares_only', False)

    if args and args.frame is not None:
        start_override = int(args.frame)
        end_exclusive_override = int(args.frame) + 1
    elif args and args.end_frame is not None and end_exclusive_override is None:
        end_exclusive_override = int(args.end_frame) + 1

    try:
        while not stop_requested[0]:
            # Load records and filter targets
            records = load_upload_records(records_path)
            target_records = [
                r for r in records
                if r.get("annotation_progress", 0) == 3.0
            ]

            if target_records:
                empty_poll_count = 0
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{TASK_NAME}] [{ts}] Found {len(target_records)} tasks to process")

                progress_reporter.start(total=len(target_records), message=f"Found {len(target_records)} tasks")

                records_dir = records_path.parent

                for idx, rec in enumerate(target_records):
                    if stop_requested[0]:
                        break

                    sf = rec.get("session_folder", "")
                    if not sf:
                        print(f"[Skip] Record missing session_folder")
                        continue

                    video_dir = os.path.normpath(os.path.join(records_dir, sf))
                    if not os.path.isdir(video_dir):
                        print(f"[Skip] Directory not found: {video_dir}")
                        continue

                    progress_reporter.update(idx + 1, item_name=sf, message=f"Optimizing: {sf}")
                    print(f"[{TASK_NAME}] Optimizing: {video_dir}")

                    try:
                        ok = optimize_single_record(
                            video_dir,
                            max_frames=max_frames,
                            save_ls_meshes=save_ls_meshes,
                            ls_mesh_dir=ls_mesh_dir,
                            start_frame_override=start_override,
                            end_frame_exclusive_override=end_exclusive_override,
                            use_least_squares_only=use_least_squares_only,
                        )
                        if ok:
                            atomic_update_progress(records_path, sf, 4.0)
                            print(f"[OK] Updated progress -> 4.0: {sf}")
                        else:
                            print(f"[WARN] Optimization returned False for: {sf}")
                    except Exception as e:
                        print(f"[ERROR] {video_dir}: {e}")
                        import traceback
                        traceback.print_exc()

                progress_reporter.complete(f"Completed {len(target_records)} tasks")

                if stop_requested[0]:
                    break
            else:
                empty_poll_count += 1
                # Adaptive logging
                if empty_poll_count <= 3 or empty_poll_count % 10 == 0:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{TASK_NAME}] [{ts}] No pending tasks, waiting {poll_interval}s...")

            # Wait for next poll (interruptible)
            for _ in range(poll_interval):
                if stop_requested[0]:
                    break
                time.sleep(1)

    except KeyboardInterrupt:
        pass

    print(f"\n[{TASK_NAME}] Daemon exited")


# ============= Main Entry Point =============

def main():
    parser = argparse.ArgumentParser(description="HOI Optimization Pipeline")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Session folder (single-video mode). Required unless --batch or --daemon.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: process all records with annotation_progress==3.0, then exit.",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Daemon mode: continuously poll for new tasks (annotation_progress==3.0).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=120,
        help="Poll interval in seconds for daemon mode (default: 120)",
    )
    parser.add_argument(
        "--upload_records",
        type=str,
        default=None,
        help=f"Path to upload_records.json. Default: {DEFAULT_UPLOAD_RECORDS}",
    )
    parser.add_argument("--max_frames", type=int, default=None, help="Debug: only optimize the first N frames from start_frame_index")
    parser.add_argument("--start_frame", type=int, default=None, help="Optimize from this absolute frame index (clamped to merged.start_frame_index)")
    parser.add_argument("--end_frame", type=int, default=None, help="Optimize until this absolute frame index (inclusive)")
    parser.add_argument("--end_frame_exclusive", type=int, default=None, help="Optimize until this absolute frame index (exclusive)")
    parser.add_argument("--frame", type=int, default=None, help="Optimize a single absolute frame index (sets start_frame=frame, end_frame_exclusive=frame+1)")
    parser.add_argument("--save_ls_meshes", action="store_true", help="Save human/object meshes after least-squares in HOISolver")
    parser.add_argument("--ls_mesh_dir", default="debug_ls_meshes", help="Subdir under session folder to save debug meshes")
    parser.add_argument("--use_least_squares_only", action="store_true", help="Use only least squares solver (skip human IK and Adam optimization)")
    parser.add_argument("--best_frame", type=int, default=None, help="Specify best frame (absolute index) for static object optimization. If not set, auto-detect from annotations.")
    args = parser.parse_args()

    # Resolve upload_records path
    records_path = Path(args.upload_records) if args.upload_records else DEFAULT_UPLOAD_RECORDS

    # Frame range overrides
    start_override = args.start_frame
    end_exclusive_override = args.end_frame_exclusive
    if args.frame is not None:
        start_override = int(args.frame)
        end_exclusive_override = int(args.frame) + 1
    elif args.end_frame is not None and end_exclusive_override is None:
        end_exclusive_override = int(args.end_frame) + 1

    # Daemon mode
    if args.daemon:
        if not records_path.is_file():
            print(f"[ERROR] upload_records not found: {records_path}")
            print(f"[INFO] Create it or specify --upload_records <path>")
            sys.exit(1)
        daemon_loop(
            records_path=records_path,
            poll_interval=args.poll_interval,
            args=args,
        )
        return

    # Batch mode
    if args.batch:
        if not records_path.is_file():
            raise FileNotFoundError(f"upload_records not found: {records_path}")
        records = load_upload_records(records_path)
        target = [r for r in records if r.get("annotation_progress") == 3.0]
        if not target:
            print("No records with annotation_progress == 3.0")
            return
        records_dir = records_path.parent
        for rec in target:
            sf = rec.get("session_folder", "")
            if not sf:
                print("[Skip] Record missing session_folder")
                continue
            video_dir = os.path.normpath(os.path.join(records_dir, sf))
            if not os.path.isdir(video_dir):
                print(f"[Skip] Directory not found: {video_dir}")
                continue
            print(f"Optimizing: {video_dir}")
            try:
                ok = optimize_single_record(
                    video_dir,
                    max_frames=args.max_frames,
                    save_ls_meshes=args.save_ls_meshes,
                    ls_mesh_dir=args.ls_mesh_dir,
                    start_frame_override=start_override,
                    end_frame_exclusive_override=end_exclusive_override,
                    use_least_squares_only=args.use_least_squares_only,
                    best_frame=args.best_frame,
                )
                if ok:
                    atomic_update_progress(records_path, sf, 4.0)
                    print(f"[OK] Updated progress -> 4.0: {sf}")
            except Exception as e:
                print(f"[ERROR] {video_dir}: {e}")
                raise
        return

    # Single video mode
    if not args.data_dir:
        parser.error("--data_dir is required when not using --batch or --daemon")
    optimize_single_record(
        args.data_dir,
        max_frames=args.max_frames,
        save_ls_meshes=args.save_ls_meshes,
        ls_mesh_dir=args.ls_mesh_dir,
        start_frame_override=start_override,
        end_frame_exclusive_override=end_exclusive_override,
        use_least_squares_only=args.use_least_squares_only,
        best_frame=args.best_frame,
    )


if __name__ == "__main__":
    main()
