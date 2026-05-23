# InterPoint

Interaction point prediction model for 4D Human-Object Interaction. Given an RGB image and a 3D object point cloud, the model predicts which human body joints are in contact with the object and where on the object surface those contacts occur.

## Architecture

```
RGB Image (224x224)          Object Point Cloud (1024x3)
       |                              |
  [VLM Encoder]               [PointNet++ Encoder]
       |                              |
       +----------+-------------------+
                  |
      [Interaction Transformer]
         Cross-modal attention
                  |
          +-------+-------+
          |               |
   [Human Head]    [Object Head]
   87-way binary   87x3 coordinates
   classification  on object surface
```

- **VLM Encoder**: LLaVA-based vision-language model for semantic features
- **PointNet++ Encoder**: 3D point cloud feature extraction
- **Interaction Transformer**: Cross-attention fusion of image and geometry features
- **Human Head**: Predicts contact probability for 87 SMPL-X body keypoints
- **Object Head**: Predicts 3D coordinates on the object surface for each contact

## Project Structure

```
interpoint/
├── models/                    # Model architecture
│   ├── ivd_model.py           # Main model (IVDModel)
│   ├── vlm_module.py          # VLM encoder (LLaVA)
│   ├── interaction_transformer.py  # Cross-modal transformer
│   ├── pointnet2_encoder.py   # PointNet++ encoder
│   ├── pointnet2_decoder.py   # Feature decoder
│   └── losses.py              # Loss functions
│
├── data/                      # Dataset and transforms
│   ├── transforms.py          # Image preprocessing
│   └── part_kp.json           # 87 SMPL-X keypoint definitions
│
├── utils/                     # Utilities
│   ├── keypoints.py           # Keypoint manager
│   └── metrics.py             # Evaluation metrics
│
├── configs/
│   └── default.yaml           # Model configuration
│
└── requirements.txt           # Dependencies
```

## Quick Start

### Install

```bash
pip install -r requirements.txt
# pytorch3d must be built from source, see https://github.com/facebookresearch/pytorch3d
```

### Download Pretrained Checkpoints

Download the released InterPoint checkpoint from Hugging Face and place it under `checkpoints/`:

| Checkpoint | Description | Link |
|------------|-------------|------|
| `4dhoi_contrastive_from_scratch/epoch_078.pth` | Trained on the Open4DHOI release | [Hugging Face](https://huggingface.co/datasets/acane2/Open4DHOI/blob/main/checkpoints/4dhoi_contrastive_from_scratch/epoch_078.pth) |

```bash
mkdir -p checkpoints
# Recommended placement (used by current annotator config):
# checkpoints/epoch_078.pth

# Optional backward-compatible placement:
# checkpoints/4dhoi_contrastive_from_scratch/epoch_078.pth
```

### Inference

The model is integrated into the [4dhoi_annotator](../4dhoi_annotator/) as an optional auto-prediction feature. To use it:

1. Place the checkpoint in `checkpoints/`
2. In `4dhoi_annotator/config.yaml`, set:
   ```yaml
   ivd_model:
     enabled: true
   ```
3. The annotator will load the model at startup and provide an "Auto Predict" button

You can also run inference directly:

```python
from ivd_predictor import IVDPredictor

predictor = IVDPredictor(checkpoint_path="checkpoints/4dhoi_contrastive_from_scratch/epoch_078.pth")
predictions = predictor.predict(rgb_frame, object_vertices, threshold=0.3)
# Returns: [{'joint': 'left_hand', 'xyz': [x,y,z], 'confidence': 0.85}, ...]
```

You can also use:

```python
predictor = IVDPredictor(checkpoint_path="checkpoints/epoch_078.pth")
```

## Configuration

Model hyperparameters in `configs/default.yaml`:
- `d_tr`: Transformer dimension (256)
- `num_body_points`: Number of human keypoints (87)
- `num_object_queries`: Number of object surface queries (87)

## License

[TODO: Add license]

## Citation

[TODO: Add citation]
