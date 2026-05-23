# -*- coding: utf-8 -*-

import os
# Respect launcher-provided GPU assignment (e.g., CUDA_VISIBLE_DEVICES from step script).
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from task_status import TaskProgress

PROJECT_DIR = Path(__file__).resolve().parent
UPLOAD_RECORDS_PATH = PROJECT_DIR / "upload_records.json"


def _resolve_path(p: str) -> Path:
    if not isinstance(p, str) or not p:
        return Path("")
    if p.startswith("./"):
        return (PROJECT_DIR / p[2:]).resolve()
    if p.startswith("tiktok_data/"):
        return (PROJECT_DIR / p).resolve()
    return Path(p).expanduser().resolve()


def _load_records() -> list[dict]:
    if not UPLOAD_RECORDS_PATH.exists():
        raise FileNotFoundError(f"upload_records.json not found: {UPLOAD_RECORDS_PATH}")
    with UPLOAD_RECORDS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("upload_records.json must be a list")
    return data


def _update_record_progress(records: list[dict], session_folder: str, progress: float | int) -> None:
    for rec in records:
        if str(rec.get("session_folder", "")) == session_folder:
            rec["annotation_progress"] = progress
            return


def _write_records(records: list[dict]) -> None:
    tmp_path = UPLOAD_RECORDS_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)
    tmp_path.replace(UPLOAD_RECORDS_PATH)


def _load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_frames(frames_dir: Path):
    exts = (".png", ".jpg", ".jpeg")
    paths = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not paths:
        raise RuntimeError(f"No frames found in: {frames_dir}")

    def _key(p: Path):
        try:
            return int(p.stem)
        except Exception:
            return p.stem

    return sorted(paths, key=_key)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Demo sequence directory; if omitted, batch from upload_records (progress=1.3)",
    )
    p.add_argument(
        "--daemon",
        action="store_true",
        help="守护进程模式：持续轮询等待新任务",
    )
    p.add_argument(
        "--poll-interval",
        type=int,
        default=120,
        help="轮询间隔（秒），默认 120",
    )
    return p.parse_args()

# Resolve Depth-Anything-V2 root from ./Depth-Anything-V2 (script dir or cwd)
_SCRIPT_DIR = Path(__file__).resolve().parent
_CWD = Path.cwd().resolve()

DA_ROOT = Path(os.environ.get("DEPTH_ANYTHING_ROOT", "/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/reconstruction/Depth-Anything-V2"))

# Add repo to sys.path so `import depth_anything_v2...` works
sys.path.insert(0, str(DA_ROOT))

# Verify package import target exists (best-effort sanity check)
DA_PKG_DIR = (DA_ROOT / "depth_anything_v2").resolve()
assert DA_PKG_DIR.exists(), (
    f"Depth-Anything-V2 python package not found.\n"
    f"Expected package dir at: {DA_PKG_DIR}\n"
    f"DA_ROOT: {DA_ROOT}"
)

def _build_model():
    try:
        from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import DepthAnythingV2 from {DA_ROOT}: {repr(e)}")
    encoder = "vitl"
    ckpt_path = DA_ROOT / "checkpoints" / "depth_anything_v2_vitl.pth"
    if not ckpt_path.exists():
        raise RuntimeError(f"Depth-Anything-V2 checkpoint not found: {ckpt_path}")
    model_configs = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    model = DepthAnythingV2(**model_configs[encoder])
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state, strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, device


def _process_one(demo_dir: Path, model, device: str) -> None:
    frames_dir = demo_dir / "frames"
    select_json = demo_dir / "select_id.json"
    if not demo_dir.exists():
        raise RuntimeError(f"video_dir not found: {demo_dir}")
    if not frames_dir.exists():
        raise RuntimeError(f"frames folder not found: {frames_dir}")
    if not select_json.exists():
        raise RuntimeError(f"select_id.json not found: {select_json}")

    frame_paths = _list_frames(frames_dir)
    T = len(frame_paths)
    sel = _load_json(select_json)
    select_id = int(sel.get("select_id", 0))
    select_index = 0
    for i, fp in enumerate(frame_paths):
        try:
            if int(fp.stem) == select_id:
                select_index = i
                break
        except Exception:
            pass
    if not (0 <= select_index < T):
        select_index = 0

    depths = []
    input_size = 518
    for fp in tqdm(frame_paths, desc="depth", unit="fr"):
        bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read frame: {fp}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            d = model.infer_image(rgb, input_size=input_size)
            d = np.asarray(d, dtype=np.float32)
        depths.append(d)

    depth_arr = np.stack(depths, axis=0).astype(np.float32)
    depth_npy = demo_dir / "depth.npy"
    np.save(str(depth_npy), depth_arr)
    out_dir = demo_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_depth_npy = out_dir / "raw_depth.npy"
    np.save(str(raw_depth_npy), depth_arr[select_index])


def main():
    args = parse_args()

    # 互斥检查
    if args.daemon and args.video_dir:
        print("错误: --daemon 和 --video_dir 不能同时使用")
        sys.exit(1)

    if args.video_dir:
        demo_dir = Path(args.video_dir).expanduser().resolve()
        model, device = _build_model()
        _process_one(demo_dir, model, device)
        return

    # 守护模式
    if args.daemon:
        from daemon_runner import daemon_loop, resolve_path, atomic_update_progress

        TARGET_PROGRESS = 1.3
        NEXT_PROGRESS = 1.4

        def init_fn(args):
            model, device = _build_model()
            return {"model": model, "device": device}

        def process_batch(records, target_records, context, args, progress_reporter=None):
            model = context["model"]
            device = context["device"]

            for i, rec in enumerate(tqdm(target_records, desc="make_depth", unit="vid")):
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
                    _process_one(session_path, model, device)
                    atomic_update_progress(sf, NEXT_PROGRESS)
                    tqdm.write(f"[完成] {video_name} -> progress={NEXT_PROGRESS}")
                except Exception as e:
                    tqdm.write(f"[错误] {video_name}: {e}")
                    raise

        daemon_loop(
            task_name="make_depth",
            target_progress=TARGET_PROGRESS,
            next_progress=NEXT_PROGRESS,
            process_batch_fn=process_batch,
            init_fn=init_fn,
            poll_interval=args.poll_interval,
            args=args,
        )
        return

    # 原有批量模式
    records = _load_records()
    target = [r for r in records if r.get("annotation_progress", 0) == 1.3]
    if not target:
        return

    # 初始化进度上报
    progress = TaskProgress("make_depth")
    progress.start(total=len(target), message=f"开始处理 {len(target)} 个视频")

    model, device = _build_model()

    try:
        for i, rec in enumerate(tqdm(target, desc="make_depth", unit="vid")):
            sf = rec.get("session_folder", "")
            if not sf:
                raise RuntimeError("记录缺少 session_folder")
            session_path = _resolve_path(sf)
            if not session_path.exists():
                raise RuntimeError(f"目录不存在: {session_path}")
            video_name = rec.get("file_name", Path(sf).name)
            object_category = rec.get("object_category", "未知")

            # 上报当前进度
            progress.update(i + 1, item_name=video_name, message=f"正在处理: {video_name}")

            tqdm.write(f"视频: {video_name} | 物体: {object_category}")
            _process_one(session_path, model, device)
            _update_record_progress(records, sf, 1.4)
            _write_records(records)

        # 完成
        progress.complete(f"完成！共处理 {len(target)} 个视频")
    except Exception as e:
        progress.fail(str(e))
        raise


if __name__ == "__main__":
    main()
