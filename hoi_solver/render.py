import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # specify GPU id
import glob
import json
import numpy as np
import torch
import smplx
import argparse
from PIL import Image
from tqdm import tqdm
# renderer & utilities from project
from video_optimizer.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
import trimesh
from video_optimizer.utils.parameter_transform import compute_combined_transform, apply_transform_to_smpl_params


def _default_smpl_model_path() -> str:
    env_path = os.environ.get("SMPLX_MODEL")
    if env_path and os.path.exists(env_path):
        return env_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    shared_path = os.path.join(repo_root, "shared_data", "SMPLX_NEUTRAL.npz")
    if os.path.exists(shared_path):
        return shared_path

    # Fallback to legacy relative path for compatibility.
    return 'video_optimizer/smpl_models/SMPLX_NEUTRAL.npz'


def _extract_yyyymmdd_hhmmss(filepath: str):
    """Extract datetime key from filenames like *_YYYYMMDD_HHMMSS.json.

    We prefer filename timestamps over filesystem mtime because files may be copied
    or touched, making mtime unreliable for determining which run is newer.
    """
    import re
    base = os.path.basename(filepath)
    m = re.search(r"_(\d{8})_(\d{6})(?:\.[^.]+)?$", base)
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}"

def find_latest_file(dirpath, pattern):
    files = glob.glob(os.path.join(dirpath, pattern))
    if not files:
        return None

    # Prefer timestamps embedded in filenames (e.g. transformed_parameters_YYYYMMDD_HHMMSS.json)
    # to avoid mtime being skewed by copy/rsync/git operations.
    keys = [(f, _extract_yyyymmdd_hhmmss(f)) for f in files]
    ts_keys = [(f, k) for (f, k) in keys if k is not None]
    if ts_keys:
        # Lexicographic order works for YYYYMMDD_HHMMSS.
        return max(ts_keys, key=lambda x: x[1])[0]

    return max(files, key=os.path.getmtime)

def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

def ensure_cuda(t):
    return t.cuda() if torch.cuda.is_available() else t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='session folder containing final_optimized_parameters')
    parser.add_argument('--params_json', default=None, help='optional explicit parameters json (e.g. final_optimized_parameters/all_parameters_*.json)')
    parser.add_argument('--use_transformed_object_params', action='store_true', help='use embedded transformed object params (R_total/T_total) if present; default is to recompute for consistency')
    parser.add_argument('--save_transformed_params', action='store_true', help='save cam->global transformed params JSON (human/object) in transformed_parameters_final.json style')
    parser.add_argument('--transformed_out', default=None, help='output path for transformed parameters json; default: <data_dir>/final_optimized_parameters/transformed_parameters_final.json')
    parser.add_argument('--ground_align', choices=['miny', 'none'], default='none', help='how to align scene with y=0 ground plane (default: none)')
    parser.add_argument('--smpl_model', default=_default_smpl_model_path())
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--out', default='output_render.mp4')
    parser.add_argument('--start_frame', type=int, default=None, help='optional absolute start frame (inclusive)')
    parser.add_argument('--end_frame_exclusive', type=int, default=None, help='optional absolute end frame (exclusive)')
    args = parser.parse_args()

    fop_dir = os.path.join(args.data_dir, 'final_optimized_parameters')
    if not os.path.isdir(fop_dir):
        raise FileNotFoundError(f"Not found: {fop_dir}")

    if args.params_json is not None:
        combined_json = args.params_json
        if not os.path.exists(combined_json):
            raise FileNotFoundError(f"params_json not found: {combined_json}")
    else:
        latest_path = os.path.join(fop_dir, "all_parameters_latest.json")
        if os.path.isfile(latest_path):
            combined_json = latest_path
        else:
            combined_json = find_latest_file(fop_dir, "all_parameters_*.json") or find_latest_file(fop_dir, "transformed_parameters_*.json")
        if not combined_json:
            raise FileNotFoundError("No all_parameters_latest.json or all_parameters_*.json / transformed_parameters_*.json found in final_optimized_parameters")

    with open(combined_json, 'r', encoding='utf-8') as f:
        combined = json.load(f)
    human_raw = combined['human_params_raw']
    object_raw = combined['object_params_raw']
    incam_subset = combined['smpl_params_incam_subset']
    global_subset = combined['smpl_params_global_subset']
    transformed_section = combined['transformed']
    cam_info = combined.get('camera') or {}
    K_saved = cam_info.get('K')
    print(f"Frames in human raw: {len(human_raw.get('body_pose', {}))}, object raw: {len(object_raw.get('poses', {}))}, incam subset: {len(incam_subset.get('global_orient', {}))}, global subset: {len(global_subset.get('global_orient', {}))}")
    frame_range = (combined.get('metadata', {}) or {}).get('frame_range', {})
    base_start = int(frame_range.get('start', 0))
    base_end_exclusive = int(frame_range.get('end', base_start))

    # Always use the single baked mesh from the session.
    transformed_params = (transformed_section.get('parameters') or {}) if isinstance(transformed_section, dict) else {}
    transformed_object_params_file = (transformed_params.get('object_params') or {}) if isinstance(transformed_params, dict) else {}

    # Check if this is self_data/pass data - if so, use obj_init.obj directly without preprocessing
    is_self_data_pass = 'self_data/pass' in args.data_dir or 'self_data\\pass' in args.data_dir

    if is_self_data_pass:
        # For self_data/pass, use obj_init.obj directly without any preprocessing
        mesh_path = os.path.join(args.data_dir, 'obj_init.obj')
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(f"object mesh (.obj) not found: {mesh_path}")
        print(f"[render] Detected self_data/pass data, using obj_init.obj directly without preprocessing")
        obj_poses_data = {'scale': 1.0}
        merged_data = {'object_scale': 1.0}
    else:
        # Use obj_org.obj and apply preprocessing to match optimization
        mesh_path = os.path.join(args.data_dir, 'obj_org.obj')
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(f"object mesh (.obj) not found: {mesh_path}")

        # Load obj_poses.json to get scale
        obj_poses_path = os.path.join(args.data_dir, 'align', 'obj_poses.json')
        if not os.path.isfile(obj_poses_path):
            obj_poses_path = os.path.join(args.data_dir, 'output', 'obj_poses.json')

        if os.path.isfile(obj_poses_path):
            with open(obj_poses_path, 'r') as f:
                obj_poses_data = json.load(f)
        else:
            print(f"[render] WARN: obj_poses.json not found, using default scale=1.0")
            obj_poses_data = {'scale': 1.0}

        # Load kp_record_new.json to get object_scale
        kp_record_path = os.path.join(args.data_dir, 'kp_record_new.json')
        if os.path.isfile(kp_record_path):
            with open(kp_record_path, 'r') as f:
                merged_data = json.load(f)
        else:
            print(f"[render] WARN: kp_record_new.json not found, using default object_scale=1.0")
            merged_data = {'object_scale': 1.0}
 
    # SMPL model
    smpl_model = smplx.create(
        args.smpl_model,
        model_type='smplx',
        gender='neutral',
        num_betas=10,
        num_expression_coeffs=10,
        use_pca=False,
        flat_hand_mean=True
    )
    if torch.cuda.is_available():
        smpl_model = smpl_model.cuda()
    frame_keys = sorted(human_raw.get('body_pose', {}).keys(), key=lambda k: int(k))
    if not frame_keys:
        raise ValueError("No frames found in human params")

    # Apply optional subrange
    start_frame = base_start if args.start_frame is None else int(args.start_frame)
    end_frame_exclusive = base_end_exclusive if args.end_frame_exclusive is None else int(args.end_frame_exclusive)
    frame_keys = [k for k in frame_keys if start_frame <= int(k) < end_frame_exclusive]
    if not frame_keys:
        raise ValueError(f"No frames in requested range: [{start_frame}, {end_frame_exclusive})")

    # Load and preprocess object mesh
    obj_tm = trimesh.load(mesh_path, process=False)
    obj_vertices_raw = np.asarray(obj_tm.vertices, dtype=np.float32)

    if is_self_data_pass:
        # For self_data/pass, use vertices directly without preprocessing
        obj_vertices = obj_vertices_raw
        print(f"[render] Using obj_init.obj directly without preprocessing (self_data/pass)")
    else:
        # Apply preprocessing: scale, recenter, object_scale (same as optimize.py)
        scale = float(obj_poses_data.get('scale', 1.0))
        obj_vertices = obj_vertices_raw * scale
        center = np.mean(obj_vertices, axis=0)
        obj_vertices = obj_vertices - center
        object_scale = float(merged_data.get('object_scale', 1.0))
        obj_vertices = obj_vertices * object_scale
        print(f"[render] Preprocessed object mesh: scale={scale}, object_scale={object_scale}")

    obj_faces = np.asarray(obj_tm.faces, dtype=np.int32)
    obj_colors = np.asarray(obj_tm.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0 if obj_tm.visual.vertex_colors is not None else None

    num_frames = len(frame_keys)
    human_vertices = []
    faces_human = None

    # Precompute cam->global transforms per frame using original incam/global relation.
    R_combined_list = []
    T_combined_list = []
    for fk in frame_keys:
        fi = int(fk)
        subset_idx = fi - base_start
        incam_o = torch.tensor(incam_subset['global_orient'][subset_idx], dtype=torch.float32)
        incam_t = torch.tensor(incam_subset['transl'][subset_idx], dtype=torch.float32)
        glob_o = torch.tensor(global_subset['global_orient'][subset_idx], dtype=torch.float32)
        glob_t = torch.tensor(global_subset['transl'][subset_idx], dtype=torch.float32)
        R_c2g, T_c2g = compute_combined_transform((incam_o, incam_t), (glob_o, glob_t))
        R_combined_list.append(np.asarray(R_c2g, dtype=np.float32))
        T_combined_list.append(np.asarray(T_c2g, dtype=np.float32))

    # Optional: save transformed (global) parameters in a standalone json.
    if args.save_transformed_params:
        transformed_human = {
            'body_pose': {},
            'betas': {},
            'global_orient': {},
            'transl': {},
            'left_hand_pose': {},
            'right_hand_pose': {},
        }
        transformed_object = {
            'R_total': {},
            'T_total': {},
        }

        for idx, fk in enumerate(frame_keys):
            fi = int(fk)
            subset_idx = fi - base_start

            # Copy pose/shape/hand params (they are frame-local and don't change under rigid world transform).
            transformed_human['body_pose'][fk] = human_raw['body_pose'][fk]
            transformed_human['betas'][fk] = human_raw['betas'][fk]
            transformed_human['left_hand_pose'][fk] = human_raw['left_hand_pose'][fk]
            transformed_human['right_hand_pose'][fk] = human_raw['right_hand_pose'][fk]

            # Transform root orient/transl from camera to global.
            incam_orient_opt = torch.tensor(human_raw['global_orient'][fk], dtype=torch.float32)
            incam_transl_opt = torch.tensor(human_raw['transl'][fk], dtype=torch.float32)

            incam_o = torch.tensor(incam_subset['global_orient'][subset_idx], dtype=torch.float32)
            incam_t = torch.tensor(incam_subset['transl'][subset_idx], dtype=torch.float32)
            glob_o = torch.tensor(global_subset['global_orient'][subset_idx], dtype=torch.float32)
            glob_t = torch.tensor(global_subset['transl'][subset_idx], dtype=torch.float32)

            new_orient, new_transl = apply_transform_to_smpl_params(
                incam_orient_opt,
                incam_transl_opt,
                (incam_o, incam_t),
                (glob_o, glob_t),
            )
            transformed_human['global_orient'][fk] = np.asarray(new_orient, dtype=np.float32).reshape(-1).tolist()
            transformed_human['transl'][fk] = np.asarray(new_transl, dtype=np.float32).reshape(-1).tolist()

            # Object params: convert camera-space R/t to global R_total/T_total.
            R_obj = np.asarray(object_raw['poses'][fk], dtype=np.float32)
            t_obj = np.asarray(object_raw['centers'][fk], dtype=np.float32)
            R_c2g = R_combined_list[idx]
            T_c2g = T_combined_list[idx]
            R_total = (R_c2g @ R_obj).astype(np.float32)
            T_total = (R_c2g @ t_obj + T_c2g).astype(np.float32)
            transformed_object['R_total'][fk] = R_total.tolist()
            transformed_object['T_total'][fk] = T_total.tolist()

        out_path = args.transformed_out
        if out_path is None:
            out_path = os.path.join(fop_dir, 'transformed_parameters_final.json')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        payload = {
            'human_params_transformed': transformed_human,
            'object_params_transformed': transformed_object,
        }
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
        print(f"[render] Saved transformed parameters: {out_path}")

    for idx, fk in tqdm(enumerate(frame_keys), desc='SMPL forward'):
        bp = np.asarray(human_raw['body_pose'][fk], dtype=np.float32)
        betas = np.asarray(human_raw['betas'][fk], dtype=np.float32)
        left_hand = np.asarray(human_raw['left_hand_pose'][fk], dtype=np.float32)
        right_hand = np.asarray(human_raw['right_hand_pose'][fk], dtype=np.float32)

        # IMPORTANT: use optimized incam params from human_raw (matches preview).
        incam_orient_opt = np.asarray(human_raw['global_orient'][fk], dtype=np.float32)
        incam_transl_opt = np.asarray(human_raw['transl'][fk], dtype=np.float32)
        with torch.no_grad():
            bpt = ensure_cuda(torch.tensor(bp).view(1, -1))
            betat = ensure_cuda(torch.tensor(betas).view(1, -1))
            glt = ensure_cuda(torch.tensor(incam_orient_opt).view(1, 3))
            trt = ensure_cuda(torch.tensor(incam_transl_opt).view(1, 3))
            lht = ensure_cuda(torch.tensor(left_hand).view(1, -1))
            rht = ensure_cuda(torch.tensor(right_hand).view(1, -1))
            out = smpl_model(
                betas=betat,
                body_pose=bpt,
                left_hand_pose=lht,
                right_hand_pose=rht,
                jaw_pose=torch.zeros((1,3)).cuda() if torch.cuda.is_available() else torch.zeros((1,3)),
                leye_pose=torch.zeros((1,3)).cuda() if torch.cuda.is_available() else torch.zeros((1,3)),
                reye_pose=torch.zeros((1,3)).cuda() if torch.cuda.is_available() else torch.zeros((1,3)),
                global_orient=glt,
                expression=torch.zeros((1,10)).cuda() if torch.cuda.is_available() else torch.zeros((1,10)),
                transl=trt
            )
        verts_cam = out.vertices[0].cpu().numpy().astype(np.float32)
        # Map optimized camera-space vertices into global for turntable rendering.
        R_c2g = R_combined_list[idx]
        T_c2g = T_combined_list[idx]
        verts_glob = (verts_cam @ R_c2g.T) + T_c2g
        human_vertices.append(verts_glob)
        if faces_human is None:
            faces_human = np.asarray(smpl_model.faces, dtype=np.int32)
    human_vertices_transformed = []
    object_vertices_per_frame = []
    for idx, fk in enumerate(frame_keys):
        hverts = human_vertices[idx]
        # Compute object vertices in camera space from optimized raw params
        R_obj = np.asarray(object_raw['poses'][fk], dtype=np.float32)
        t_obj = np.asarray(object_raw['centers'][fk], dtype=np.float32)

        if args.use_transformed_object_params and isinstance(transformed_object_params_file, dict) and 'R_total' in transformed_object_params_file and fk in transformed_object_params_file['R_total']:
            # Use precomputed global transform if explicitly requested
            R_total = np.asarray(transformed_object_params_file['R_total'][fk], dtype=np.float32)
            T_total = np.asarray(transformed_object_params_file['T_total'][fk], dtype=np.float32)
            overts_glob = (obj_vertices @ R_total.T) + T_total
        else:
            # Map camera-space object into global using same cam->global used for human
            overts_cam = (obj_vertices @ R_obj.T) + t_obj
            R_c2g = R_combined_list[idx]
            T_c2g = T_combined_list[idx]
            overts_glob = (overts_cam @ R_c2g.T) + T_c2g

        human_vertices_transformed.append(hverts.astype(np.float32))
        object_vertices_per_frame.append(overts_glob.astype(np.float32))

    verts_human_t = torch.tensor(np.stack(human_vertices_transformed, axis=0), dtype=torch.float32).cuda()
    verts_object_t = torch.tensor(np.stack(object_vertices_per_frame, axis=0), dtype=torch.float32).cuda()
    faces_human_t = faces_human
    faces_obj = obj_faces
    if isinstance(K_saved, (list, tuple)) and len(K_saved) == 3:
        K = torch.tensor(K_saved, dtype=torch.float32)
    else:
        K = torch.tensor([
            [1200.0, 0.0, args.width / 2],
            [0.0, 1200.0, args.height / 2],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)
    if torch.cuda.is_available():
        K = K.cuda()

    renderer = Renderer(args.width, args.height, device="cuda" if torch.cuda.is_available() else "cpu",
                        faces_human=faces_human_t, faces_obj=faces_obj, K=K)
    # Optional: align the whole scene to the y=0 ground plane.
    if args.ground_align == 'miny':
        combined_for_min = torch.cat([verts_human_t, verts_object_t], dim=1)
        y_min = combined_for_min[..., 1].min()
        # Shift only along y; keep x/z unchanged.
        shift = torch.tensor([0.0, y_min.item(), 0.0], dtype=verts_human_t.dtype, device=verts_human_t.device)
        verts_human_t = verts_human_t - shift
        verts_object_t = verts_object_t - shift
        print(f"[render] ground_align=miny: shifted scene down by y={y_min.item():.6f}")

    combined_verts = torch.cat([verts_human_t, verts_object_t], dim=1)
    verts_glob = combined_verts
    

    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob.cpu(),
        beta=2.5,
        cam_height_degree=20,
        target_center_height=1.0,
        vec_rot=180
    )

    joints_glob = verts_human_t.mean(dim=1, keepdim=True)  # (F,1,3)
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob.cpu())
    renderer.set_ground(scale * 1.5, cx, cz)
    temp_dir = os.path.join(args.data_dir, 'final_optimized_parameters', 'temp_frames_render')
    os.makedirs(temp_dir, exist_ok=True)
    frame_paths = []
    color = torch.ones(3).float().cuda() * 0.8
    for i in tqdm(range(num_frames), desc='Rendering frames'):
        cams = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground_hoi(verts_human_t[i], verts_object_t[i], cams, global_lights, [0.8, 0.8, 0.8],
                                             obj_colors)
        img = np.clip(img, 0, 255).astype(np.uint8)
        path = os.path.join(temp_dir, f'frame_{i:04d}.png')
        Image.fromarray(img).save(path, optimize=False)
        frame_paths.append(path)
    out_path = os.path.join(args.data_dir, args.out)
    try:
        import importlib
        imageio = importlib.import_module('imageio.v2')

        writer = imageio.get_writer(
            out_path,
            fps=args.fps,
            format='FFMPEG',
            codec='libx264',
            ffmpeg_params=['-pix_fmt', 'yuv420p', '-crf', '18'],
        )
        try:
            for p in tqdm(frame_paths, desc='Writing video'):
                img = imageio.imread(p)
                writer.append_data(img)
        finally:
            writer.close()
    except Exception as e:
        # Fallback: write video using OpenCV if imageio/ffmpeg plugin is unavailable.
        print(f"[render] imageio ffmpeg unavailable ({e}); fallback to OpenCV writer.")
        import importlib
        cv2 = importlib.import_module('cv2')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        first = cv2.imread(frame_paths[0])
        if first is None:
            raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")
        h, w = first.shape[:2]
        vw = cv2.VideoWriter(out_path, fourcc, float(args.fps), (w, h))
        try:
            for p in tqdm(frame_paths, desc='Writing video (cv2)'):
                fr = cv2.imread(p)
                if fr is None:
                    continue
                vw.write(fr)
        finally:
            vw.release()

    print("Render complete. Video saved to:", out_path)


if __name__ == "__main__":
    main()
