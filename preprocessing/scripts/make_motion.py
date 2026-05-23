# -*- coding: utf-8 -*-
import os
# Respect launcher-provided GPU assignment (e.g., CUDA_VISIBLE_DEVICES from step script).
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import os.path
import sys
import json
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path

# Resolve GVHMR root: env var > ./GVHMR relative to script > ./GVHMR relative to cwd
_SCRIPT_DIR = Path(__file__).resolve().parent
_CWD = Path.cwd().resolve()

_env_gvhmr = os.environ.get("GVHMR_ROOT")
if _env_gvhmr and Path(_env_gvhmr).exists():
    GVHMR_ROOT = Path(_env_gvhmr).resolve()
elif (_SCRIPT_DIR / "GVHMR").exists():
    GVHMR_ROOT = (_SCRIPT_DIR / "GVHMR").resolve()
else:
    GVHMR_ROOT = (_CWD / "GVHMR").resolve()

assert GVHMR_ROOT.exists(), (
    f"GVHMR not found.\n"
    f"Set GVHMR_ROOT env var, or place GVHMR/ next to this script.\n"
    f"Checked: $GVHMR_ROOT={_env_gvhmr}, {_SCRIPT_DIR / 'GVHMR'}, {_CWD / 'GVHMR'}"
)

# ultralytics yolov8x.pt is a pickle checkpoint, blocked by weights_only=True default.
# We force weights_only=False by default BEFORE Tracker/YOLO is imported.
_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat  # type: ignore

# Make GVHMR importable
if str(GVHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GVHMR_ROOT))

PROJECT_DIR = _SCRIPT_DIR
UPLOAD_RECORDS_PATH = PROJECT_DIR / "upload_records.json"


def _resolve_path(p: str) -> Path:
    if not isinstance(p, str) or not p:
        return Path("")
    if p.startswith("./"):
        return (PROJECT_DIR / p[2:]).resolve()
    if p.startswith("tiktok_data/"):
        return (PROJECT_DIR / p).resolve()
    return Path(p).expanduser().resolve()


def _load_records() -> list:
    if not UPLOAD_RECORDS_PATH.exists():
        raise FileNotFoundError(f"upload_records.json not found: {UPLOAD_RECORDS_PATH}")
    with UPLOAD_RECORDS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("upload_records.json must be a list")
    return data


def _update_record_progress(records: list, session_folder: str, progress: float) -> None:
    for rec in records:
        if str(rec.get("session_folder", "")) == session_folder:
            rec["annotation_progress"] = progress
            return


def _write_records(records: list) -> None:
    tmp_path = UPLOAD_RECORDS_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)
    tmp_path.replace(UPLOAD_RECORDS_PATH)

# 导入进度上报模块
from task_status import TaskProgress

from hmr4d.utils.pylogger import Log
import hydra
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from pytorch3d.transforms import quaternion_to_matrix

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    read_video_np,
    save_video,
    merge_videos_horizontal,
    get_writer,
    get_video_reader,
)
from hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco17_skeleton_batch

from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SimpleVO

from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, convert_K_to_K4, create_camera_sensor
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from tqdm import tqdm
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from einops import einsum

CRF = 23  # 17 is lossless, every +6 halves the mp4 size


def _build_cfg_for_video_dir(
    video_dir: str,
    *,
    static_cam: bool = False,
    verbose: bool = False,
    use_dpvo: bool = False,
    f_mm: int | None = None,
    output_root: str | None = None,
):
    # 清理 Hydra 全局状态，避免多次调用时出错
    GlobalHydra.instance().clear()

    video_path = Path(video_dir) / "video.mp4"

    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input] {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")

    cfg_dir = (GVHMR_ROOT / "hmr4d" / "configs").resolve()
    assert cfg_dir.exists(), f"GVHMR config dir not found: {cfg_dir}"

    with initialize_config_dir(version_base="1.3", config_dir=str(cfg_dir)):
        overrides = [
            f"video_name={Path(video_dir).resolve().name}",
            f"static_cam={static_cam}",
            f"verbose={verbose}",
            f"use_dpvo={use_dpvo}",
            f"+video_dir={video_dir}",
            f"video_path={str(video_path)}",
        ]
        if f_mm is not None:
            overrides.append(f"f_mm={f_mm}")
        if output_root is not None:
            overrides.append(f"output_root={output_root}")

        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def parse_args_to_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default=None, help="By default uses GVHMR config output_root")
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip VO (slam)")
    parser.add_argument("--use_dpvo", action="store_true", help="If set, use DPVO. Default: use SimpleVO.")
    parser.add_argument("--f_mm", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="If true, draw intermediate results")
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Folder containing video.mp4; if omitted, batch from upload_records (progress=1.4)",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="守护进程模式：持续轮询等待新任务",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=120,
        help="轮询间隔（秒），默认 120",
    )
    args = parser.parse_args()
    if not args.video_dir:
        return None, args
    cfg = _build_cfg_for_video_dir(
        args.video_dir,
        static_cam=args.static_cam,
        verbose=args.verbose,
        use_dpvo=args.use_dpvo,
        f_mm=args.f_mm,
        output_root=args.output_root,
    )
    return cfg, args


def _clear_preprocess_cache(cfg):
    """删除已有预处理结果，确保每次都重新跑。"""
    paths = cfg.paths
    to_remove = [
        paths.bbx,
        paths.vitpose,
        paths.vit_features,
    ]
    if not cfg.static_cam:
        to_remove.append(paths.slam)
    for p in to_remove:
        f = Path(p)
        if f.exists():
            f.unlink()
            Log.info(f"[Preprocess] 已删除缓存: {p}")


@torch.no_grad()
def run_preprocess(cfg):
    Log.info("[Preprocess] Start!")
    tic = Log.time()
    _clear_preprocess_cache(cfg)
    video_path = cfg.video_path
    paths = cfg.paths
    static_cam = cfg.static_cam
    verbose = cfg.verbose

    # Get bbx tracking result
    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()  # (L, 4)
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (L, 3)
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx)["bbx_xys"]
        Log.info(f"[Preprocess] bbx (xyxy, xys) from {paths.bbx}")

    if verbose:
        video = read_video_np(video_path)
        bbx_xyxy = torch.load(paths.bbx)["bbx_xyxy"]
        video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, video)
        save_video(video_overlay, cfg.paths.bbx_xyxy_video_overlay)

    # Get VitPose
    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(video_path, bbx_xys)
        torch.save(vitpose, paths.vitpose)
        del vitpose_extractor
    else:
        vitpose = torch.load(paths.vitpose)
        Log.info(f"[Preprocess] vitpose from {paths.vitpose}")

    if verbose:
        video = read_video_np(video_path)
        video_overlay = draw_coco17_skeleton_batch(video, vitpose, 0.5)
        save_video(video_overlay, paths.vitpose_video_overlay)

    # Get vit features
    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor
    else:
        Log.info(f"[Preprocess] vit_features from {paths.vit_features}")

    # Get visual odometry results
    if not static_cam:
        if not Path(paths.slam).exists():
            if not cfg.use_dpvo:
                simple_vo = SimpleVO(cfg.video_path, scale=0.5, step=8, method="sift", f_mm=cfg.f_mm)
                vo_results = simple_vo.compute()
                torch.save(vo_results, paths.slam)
            else:
                from hmr4d.utils.preproc.slam import SLAMModel

                length, width, height = get_video_lwh(cfg.video_path)
                K_fullimg = estimate_K(width, height)
                intrinsics = convert_K_to_K4(K_fullimg)
                slam = SLAMModel(video_path, width, height, intrinsics, buffer=4000, resize=0.5)
                bar = tqdm(total=length, desc="DPVO")
                while True:
                    ret = slam.track()
                    if ret:
                        bar.update()
                    else:
                        break
                slam_results = slam.process()
                torch.save(slam_results, paths.slam)
        else:
            Log.info(f"[Preprocess] slam results from {paths.slam}")

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time() - tic:.2f}s")


def load_data_dict(cfg):
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
        T_w2c = torch.zeros(length, 3)
    else:
        traj = torch.load(cfg.paths.slam)
        if cfg.use_dpvo:
            traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
            R_w2c = quaternion_to_matrix(traj_quat).mT
            T_w2c = torch.from_numpy(traj[:, :3])
        else:
            R_w2c = torch.from_numpy(traj[:, :3, :3])
            T_w2c = torch.from_numpy(traj[:, :3, 3])

    if cfg.f_mm is not None:
        K_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2].repeat(length, 1, 1)
    else:
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
        "kp2d": torch.load(paths.vitpose),
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": torch.load(paths.vit_features),
        "R_w2c": R_w2c,
        "T_w2c": T_w2c,
    }
    return data


def render_global(cfg, global_path, result_path):
    global_video_path = Path(global_path)

    pred = torch.load(result_path)
    smplx = make_smplx("supermotion").cuda()

    smplx2smpl_path = (GVHMR_ROOT / "hmr4d" / "utils" / "body_model" / "smplx2smpl_sparse.pt").resolve()
    J_reg_path = (GVHMR_ROOT / "hmr4d" / "utils" / "body_model" / "smpl_neutral_J_regressor.pt").resolve()

    smplx2smpl = torch.load(str(smplx2smpl_path)).cuda()
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load(str(J_reg_path)).cuda()

    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    def move_to_start_point_face_z(verts):
        verts = verts.clone()
        offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]
        offset[1] = verts[:, :, [1]].min()
        verts = verts - offset
        T_ay2ayfz = compute_T_ayfz2ay(einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
        verts = apply_T_on_points(verts, T_ay2ayfz)
        return verts

    verts_glob = move_to_start_point_face_z(pred_ay_verts)
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")

    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob.cpu(),
        beta=2.0,
        cam_height_degree=0,
        target_center_height=1.0,
    )
    pred["global_R"] = global_R.cpu()
    pred["global_T"] = global_T.cpu()
    torch.save(pred, result_path)

    length, width, height = get_video_lwh(cfg.video_path)
    _, _, K = create_camera_sensor(width, height, 24)

    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    writer = get_writer(global_video_path, fps=30, crf=CRF)
    for i in tqdm(range(length), desc="Rendering Global"):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(verts_glob[[i]], color[None], cameras, global_lights)
        writer.write_frame(img)
    writer.close()


def _run_one(cfg, args):
    motion_output_dir = str(Path(cfg.video_dir) / "motion")
    if not os.path.exists(motion_output_dir):
        os.makedirs(motion_output_dir)
    run_preprocess(cfg)
    data = load_data_dict(cfg)
    result_path = os.path.join(motion_output_dir, "result.pt")
    global_path = os.path.join(motion_output_dir, "global.mp4")
    Log.info("[HMR4D] Predicting")
    model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()
    tic = Log.sync_time()
    pred = model.predict(data, static_cam=cfg.static_cam)
    pred = detach_to_cpu(pred)
    Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s")
    torch.save(pred, result_path)
    render_global(cfg, global_path, result_path)


if __name__ == "__main__":
    os.chdir(str(GVHMR_ROOT))
    cfg, args = parse_args_to_cfg()
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")

    # 互斥检查
    if args.daemon and args.video_dir:
        print("错误: --daemon 和 --video_dir 不能同时使用")
        sys.exit(1)

    if cfg is not None:
        _run_one(cfg, args)
        sys.exit(0)

    # 守护模式
    if args.daemon:
        from daemon_runner import daemon_loop, resolve_path, atomic_update_progress

        TARGET_PROGRESS = 1.4
        NEXT_PROGRESS = 1.5

        def process_batch(records, target_records, context, args, progress_reporter=None):
            for i, rec in enumerate(tqdm(target_records, desc="human_motion", unit="vid")):
                sf = rec.get("session_folder", "")
                if not sf:
                    tqdm.write(f"[跳过] 记录缺少 session_folder")
                    continue
                session_path = resolve_path(sf)
                if not session_path.exists():
                    tqdm.write(f"[跳过] 目录不存在: {session_path}")
                    continue
                video_name = rec.get("file_name", Path(sf).name)
                object_category = rec.get("object_category", "未知")
                tqdm.write(f"视频: {video_name} | 物体: {object_category}")

                if progress_reporter:
                    progress_reporter.update(i + 1, item_name=video_name)

                try:
                    cfg = _build_cfg_for_video_dir(
                        str(session_path),
                        static_cam=args.static_cam,
                        verbose=args.verbose,
                        use_dpvo=args.use_dpvo,
                        f_mm=args.f_mm,
                        output_root=args.output_root,
                    )
                    _run_one(cfg, args)
                    atomic_update_progress(sf, NEXT_PROGRESS)
                    tqdm.write(f"[完成] {video_name} -> progress={NEXT_PROGRESS}")
                except Exception as e:
                    tqdm.write(f"[错误] {video_name}: {e}")
                    raise

        daemon_loop(
            task_name="human_motion",
            target_progress=TARGET_PROGRESS,
            next_progress=NEXT_PROGRESS,
            process_batch_fn=process_batch,
            poll_interval=args.poll_interval,
            args=args,
        )
        sys.exit(0)

    # 原有批量模式
    records = _load_records()
    target = [r for r in records if r.get("annotation_progress", 0) == 1.4]
    if not target:
        sys.exit(0)

    # 初始化进度上报
    progress = TaskProgress("human_motion")
    progress.start(total=len(target), message=f"开始处理 {len(target)} 个视频")

    try:
        for i, rec in enumerate(tqdm(target, desc="human_motion", unit="vid")):
            sf = rec.get("session_folder", "")
            if not sf:
                raise RuntimeError("记录缺少 session_folder")
            session_path = _resolve_path(sf)
            if not session_path.exists():
                raise RuntimeError(f"目录不存在: {session_path}")
            video_name = rec.get("file_name", Path(sf).name)
            object_category = rec.get("object_category", "未知")
            tqdm.write(f"视频: {video_name} | 物体: {object_category}")

            # 上报当前进度
            progress.update(i + 1, item_name=video_name, message=f"正在处理: {video_name}")

            cfg = _build_cfg_for_video_dir(
                str(session_path),
                static_cam=args.static_cam,
                verbose=args.verbose,
                use_dpvo=args.use_dpvo,
                f_mm=args.f_mm,
                output_root=args.output_root,
            )
            _run_one(cfg, args)
            _update_record_progress(records, sf, 1.5)
            _write_records(records)

        # 完成
        progress.complete(f"完成！共处理 {len(target)} 个视频")
    except Exception as e:
        progress.fail(str(e))
        raise
