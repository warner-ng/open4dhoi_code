# 4DHOI Reconstruction Pipeline

<p align="center">
  <img src="assets/teaser.jpg" width="720" />
</p>

[[Paper (arXiv)]](https://arxiv.org/abs/2512.00960)
[[Dataset (Hugging Face)]](https://huggingface.co/datasets/acane2/Open4DHOI/tree/main)
[[Annotation Demo Video]](https://youtu.be/sssSdZNHVFc)
[[InterPoint Checkpoint]](https://huggingface.co/datasets/acane2/Open4DHOI/blob/main/checkpoints/4dhoi_contrastive_from_scratch/epoch_078.pth)

End-to-end pipeline for reconstructing 4D Human-Object Interactions from monocular video. With just **a few clicks**, preprocess a video (segmentation, motion, depth, 3D reconstruction), annotate interaction contact points, optimize poses, and render the result — **fast** and **scalable**.

## 🚀 Simplest Full Flow (from repo root)

Use this if you just want the shortest working path:

```bash
# 0) Start data_preparer and upload/split video first
conda run -n 4dhoi_pipeline python data_preparer/app.py --data_dir data --port 5020

# 1) Preprocess a saved session
bash preprocessing/run_pipeline.sh data/bike/20260522_233018_eaf586b7 --retarget --smooth 0.3

# 2) Start annotator (IMPORTANT: use this env)
# 如果没有打开，请您看下面指引
conda run -n 4dhoi_pipeline python 4dhoi_annotator/app.py --data_dir data --port 5027
```

In `data_preparer` (`http://localhost:5020`):
- Upload tab: choose category + upload video
- Click **Parse & Split Scenes**
- Select segment(s), click **Save Selected**
- Annotate tab: click human/object points and save

Then in browser:
- Open `http://localhost:5027`
- Hard refresh once (`Ctrl+Shift+R`)
- Click **Save Annotation** and **Optimize**

Finally render:

```bash
bash hoi_solver/run.sh data/bike/20260522_233018_eaf586b7 --render
```

Notes:
- Run commands from the repository root (`open4dhoi_code/`).
- If output video has only 1 frame, it usually means annotations were saved for only one frame; re-open annotator, save/optimize again, then rerun `hoi_solver/run.sh`.

If `http://localhost:5027` does not open:

```bash
# Check backend is reachable
python - <<'PY'
import urllib.request
for u in ['http://127.0.0.1:5027/','http://127.0.0.1:5027/api/hoi_tasks']:
  try:
    with urllib.request.urlopen(u, timeout=3) as r:
      print(u, 'OK', r.status)
  except Exception as e:
    print(u, 'ERR', repr(e))
PY
```

If you see `Connection refused`, the annotator process is not running (often interrupted by `Ctrl+C`). Start it again with the command above.

If `conda run -n 4dhoi_pipeline python 4dhoi_annotator/app.py --data_dir data --port 5027` looks like it has "no response":
- This is usually normal. Flask is running in the foreground and holding the terminal.
- Keep that terminal open and use another terminal for other commands.
- Open `http://localhost:5027` directly in your browser.
- Stop the server with `Ctrl+C` when you are done.

## Pipeline Overview

```
                         ┌──────────────────┐
                         │   data_preparer  │  Upload video, split scenes,
                         │   (web app)      │  annotate SAM2 point prompts
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │  preprocessing   │  Extract frames, masks, object mesh,
                         │  (shell scripts) │  human motion, depth, hand pose, HOI
                         └────────┬─────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
     ┌────────▼─────────┐ ┌──────▼───────┐  ┌────────▼─────────┐
     │ 4dhoi_annotator  │ │  interpoint  │  │    hoi_solver    │
     │ (web app)        │ │  (model)     │  │  (optimization)  │
     │                  │ │              │  │                  │
     │ 3D annotation +  │ │ Train/eval   │  │ Least-squares +  │
     │ optional auto-   │ │ contact      │  │ Adam refinement  │
     │ prediction       │ │ prediction   │  │ + rendering      │
     └──────────────────┘ └──────────────┘  └──────────────────┘
```

## Modules

| Module | Description | Entry Point |
|--------|-------------|-------------|
| [data_preparer](data_preparer/) | Upload videos, split scenes, annotate point prompts for mask generation | `python data_preparer/app.py` |
| [preprocessing](preprocessing/) | 7-step preprocessing: frames, masks, object mesh, motion, depth, hand pose, HOI assembly | `bash preprocessing/run_pipeline.sh` |
| [interpoint](interpoint/) | Interaction point prediction model (train & evaluate) | `bash interpoint/train.sh` |
| [4dhoi_annotator](4dhoi_annotator/) | Interactive 3D annotation tool with optional auto-prediction | `python 4dhoi_annotator/app.py` |
| [hoi_solver](hoi_solver/) | Contact-based pose optimization + rendering | `bash hoi_solver/run.sh` |

## Installation

See [INSTALL.md](INSTALL.md) for environment setup instructions (conda environments, dependencies, and optimization packages).

## Quick Start

### 1. Download Shared Model Files

All modules share the same SMPL-X body model files. Download from [Google Drive](https://drive.google.com/file/d/1PgASEIlFjSbMTh0x7P9-7rZ1XeMAKzHD/view?usp=sharing) and place in `shared_data/`:

```bash
# Required files in shared_data/:
shared_data/
├── SMPLX_NEUTRAL.npz           # SMPL-X neutral body model
├── J_regressor.pt              # Joint regressor weights
├── smplx_downsampling_1000.npz # Mesh downsampling matrix
└── part_kp.json                # 87 SMPL-X keypoint definitions
```

Each module symlinks to `shared_data/` automatically - no need to copy files to multiple locations.

**Download SMPL-X model**: Register at [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) and download `SMPLX_NEUTRAL.npz`.

### 2. Setup Third-Party Dependencies (for preprocessing)

```bash
# Clone SAM2, SAM-3D-Objects, GVHMR, Depth-Anything-V2, SAM-3D-Body
bash preprocessing/setup_third_party.sh
```

Then download each model's checkpoints following their READMEs. See [preprocessing/README.md](preprocessing/) for details.

### 3. Prepare a Video

```bash
# Start the data preparation web app
conda run -n 4dhoi_pipeline python data_preparer/app.py --data_dir data --port 5020
# Open http://localhost:5020
# Upload Tab: upload video → split scenes → save segments
# Annotate Tab: click human/object points → save
```

### 4. Run Preprocessing

```bash
bash preprocessing/run_pipeline.sh data/category/session_name


bash preprocessing/run_pipeline.sh data/bike/20260521_160347_7d09cae4 --retarget --smooth 0.3 --render

export CUDA_VISIBLE_DEVICES=0
export CUDA_MASKS=0
export CUDA_OBJ_ORG=0
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash preprocessing/run_pipeline.sh data/bike/20260522_171755_ebf5326f --retarget --smooth 0.3
```

This runs 7 steps across 3 conda environments:

| Step | Script | Env | Output |
|------|--------|-----|--------|
| 1.1 | Extract frames | sam3d_obj_4d | `frames/*.jpg` |
| 1.2 | Generate masks (SAM2) | sam3d_obj_4d | `mask_dir/`, `human_mask_dir/` |
| 1.3 | Reconstruct object (SAM-3D) | sam3d_obj_4d | `obj_org.obj` |
| 2.1 | Estimate motion (GVHMR) | 4dhoi_pipeline | `motion/result.pt` |
| 2.2 | Estimate depth | 4dhoi_pipeline | `depth.npy` |
| 3 | Refine hand pose (SAM-3D-Body) | mhr | `motion/result_hand.pt` |
| 4 | Assemble HOI | 4dhoi_pipeline | `output/obj_poses.json` |

### 5. Annotate Interactions

```bash
# 这行命令要在主目录下
conda activate 4dhoi_annotator
python 4dhoi_annotator/app.py --data_dir data --port 5027
# Open http://localhost:5027
# Visualize 3D human+object, click contact points, optimize
```

### 6. Run Optimization (Standalone)

```bash
bash hoi_solver/run.sh data/category/session_name --render


bash hoi_solver/run.sh data/bike/20260522_171755_ebf5326f --render

bash run.sh /home/warner/_projects/open4dhoi_code/data/bike/20260522_171755_ebf5326f --render
```

### 7. Train InterPoint Model (Optional)

```bash
cd interpoint
bash train.sh
bash evaluate.sh checkpoints/4dho_contrasive_from_scratch/epoch_078.pth
```

## Project Structure

```
4dhoi_recon_pipeline/
├── README.md
├── shared_data/                    # Shared model files (download once)
│   ├── SMPLX_NEUTRAL.npz          # SMPL-X body model
│   ├── J_regressor.pt             # Joint regressor
│   ├── smplx_downsampling_1000.npz
│   └── part_kp.json               # 87 keypoint definitions
│
├── data_preparer/                  # Video upload + point annotation web app
│   ├── app.py
│   ├── templates/index.html
│   └── static/
│
├── preprocessing/                  # Automated preprocessing pipeline
│   ├── config.sh                   # Model paths configuration
│   ├── run_pipeline.sh             # Master script
│   ├── step1_frames_masks_obj.sh   # env: sam3d_obj_4d
│   ├── step2_motion_depth.sh       # env: 4dhoi_pipeline
│   ├── step3_hand.sh               # env: mhr
│   ├── step4_hoi.sh                # env: 4dhoi_pipeline
│   └── scripts/                    # Python processing scripts
│
├── interpoint/                     # Interaction point prediction model
│   ├── models/                     # Model architecture
│   ├── data/                       # Dataset loaders
│   ├── scripts/                    # Train + evaluate
│   ├── train.sh
│   └── evaluate.sh
│
├── 4dhoi_annotator/                # Interactive 3D annotation tool
│   ├── app.py                      # Flask application
│   ├── config.yaml                 # Annotator configuration
│   ├── ivd_predictor.py            # InterPoint model wrapper
│   ├── solver/                     # Built-in optimization
│   ├── co-tracker/                 # 2D point tracking
│   └── static/
│
└── hoi_solver/                     # Standalone pose optimization
    ├── run.sh                      # One-command optimize + render
    ├── optimize.py                 # Core solver (auto-converts annotations)
    ├── render.py                   # Global-view rendering
    ├── convert_annotations.py      # Decimated → original mesh index mapping
    └── video_optimizer/            # Optimization engine
```

## Session Data Format

Each session folder follows this structure (built up progressively by the pipeline):

```
session_folder/
├── video.mp4                       # Input video
├── select_id.json                  # Frame selection (from data_preparer)
├── points.json                     # SAM2 point prompts (from data_preparer)
├── frames/                         # Extracted frames (preprocessing step 1)
├── mask_dir/                       # Object masks (preprocessing step 1)
├── human_mask_dir/                 # Human masks (preprocessing step 1)
├── obj_org.obj                     # 3D object mesh (preprocessing step 1)
├── depth.npy                       # Depth maps (preprocessing step 2)
├── motion/
│   ├── result.pt                   # SMPL-X params (preprocessing step 2)
│   └── result_hand.pt              # With refined hands (preprocessing step 3)
├── output/
│   └── obj_poses.json              # Object scale/position (preprocessing step 4)
├── kp_record_merged.json           # Contact annotations (from 4dhoi_annotator)
├── kp_record_new.json              # Converted to original mesh (auto by hoi_solver)
└── final_optimized_parameters/     # Optimization output (from hoi_solver)
    └── all_parameters_latest.json
```

## Export to ResMimic (Object + Human Motion)

This section describes how to prepare data from this repo for ResMimic motion tracking.

ResMimic HOI expects two files:

- **Object motion** (`.npz`):
  - `trans`: `(T, 3)`
  - `rot`: `(T, 4)` (quaternion)
- **Human motion** (`.pkl`):
  - `fps`, `root_pos`, `root_rot`, `dof_pos`, `local_body_pos`, `link_body_list`

### 1) Generate Open4DHOI optimized outputs

From repo root:

```bash
bash hoi_solver/run.sh data/<category>/<session_name> --render
```

Required outputs:

- `data/<category>/<session_name>/final_optimized_parameters/all_parameters_latest.json`
- `data/<category>/<session_name>/final_optimized_parameters/transformed_parameters_final.json`

### 2) Export object motion for ResMimic

Convert object trajectory from `transformed_parameters_final.json` into ResMimic format (`trans`, `rot`).

```bash
python - <<'PY'
import json, numpy as np
from pathlib import Path

session = Path('data/<category>/<session_name>')
src = session / 'final_optimized_parameters' / 'transformed_parameters_final.json'
dst = Path('/home/warner/_projects/ResMimic/assets/motions/<session_name>_object.npz')

d = json.loads(src.read_text())
obj = d['object_params_transformed']
keys = sorted(obj['T_total'].keys(), key=lambda x: int(x))
trans = np.array([obj['T_total'][k] for k in keys], dtype=np.float32)
R = np.array([obj['R_total'][k] for k in keys], dtype=np.float32)

def mat_to_quat_xyzw(m):
    q = np.empty((4,), dtype=np.float32)
    tr = m[0,0] + m[1,1] + m[2,2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q[3] = 0.25 * S
        q[0] = (m[2,1] - m[1,2]) / S
        q[1] = (m[0,2] - m[2,0]) / S
        q[2] = (m[1,0] - m[0,1]) / S
    elif (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
        S = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2
        q[3] = (m[2,1] - m[1,2]) / S
        q[0] = 0.25 * S
        q[1] = (m[0,1] + m[1,0]) / S
        q[2] = (m[0,2] + m[2,0]) / S
    elif m[1,1] > m[2,2]:
        S = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2
        q[3] = (m[0,2] - m[2,0]) / S
        q[0] = (m[0,1] + m[1,0]) / S
        q[1] = 0.25 * S
        q[2] = (m[1,2] + m[2,1]) / S
    else:
        S = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2
        q[3] = (m[1,0] - m[0,1]) / S
        q[0] = (m[0,2] + m[2,0]) / S
        q[1] = (m[1,2] + m[2,1]) / S
        q[2] = 0.25 * S
    n = np.linalg.norm(q)
    if n > 1e-8:
        q /= n
    return q

rot = np.stack([mat_to_quat_xyzw(r) for r in R], axis=0).astype(np.float32)
dst.parent.mkdir(parents=True, exist_ok=True)
np.savez(dst, trans=trans, rot=rot)
print('Saved:', dst)
print('Shapes:', trans.shape, rot.shape)
PY
```

### 3) Export human motion via GMR (G1 retarget)

G1 retarget scripts are in `/home/warner/_projects/GMR`.

Example (GVHMR output to G1 motion):

```bash
cd /home/warner/_projects/GMR
python scripts/gvhmr_to_robot.py \
  --gvhmr_pred_file <path_to_hmr4d_results.pt> \
  --robot unitree_g1 \
  --save_path outputs/<session_name>_g1.pkl
```

Or SMPL-X input:

```bash
cd /home/warner/_projects/GMR
python scripts/smplx_to_robot.py \
  --smplx_file <path_to_smplx_file> \
  --robot unitree_g1 \
  --save_path outputs/<session_name>_g1.pkl
```

### 4) Point ResMimic config to your motion files

Update `ResMimic/legged_gym/legged_gym/envs/g1/g1_hoi_bike_config.py`:

- `motion_file` -> your human `.pkl`
- `object_motion_file` -> your object `.npz`

Then run train/eval as usual in ResMimic.

### Important compatibility note

ResMimic's loader requires **non-empty** `local_body_pos` and `link_body_list` in human `.pkl`.

- The current `GMR` demo scripts (`smplx_to_robot.py`, `gvhmr_to_robot.py`) save `local_body_pos=None` and `link_body_list=None` by default.
- If this causes loading failure in ResMimic, either:
  1. Use an existing ResMimic-compatible human `.pkl` as template/baseline, or
  2. Add a post-processing step that fills `local_body_pos` and `link_body_list` (e.g., FK-based body position export for the G1 links expected by your ResMimic task).

For this reason, object trajectory export from Open4DHOI is direct, while human motion integration may require one additional retarget/postprocess step depending on your ResMimic branch.

## Citation

```bibtex
@misc{wen2026efficientscalablemonocularhumanobject,
      title={Efficient and Scalable Monocular Human-Object Interaction Motion Reconstruction},
      author={Boran Wen and Ye Lu and Sirui Wang and Keyan Wan and Jiahong Zhou and Junxuan Liang and Xinpeng Liu and Bang Xiao and Ruiyang Liu and Yong-Lu Li},
      year={2026},
      eprint={2512.00960},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.00960},
}
```
