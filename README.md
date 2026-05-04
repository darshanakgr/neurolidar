# NeuroLiDAR: Adaptive Frame Rate Depth Sensing via Neuromorphic Event-LiDAR Fusion

**[ICRA 2026]** &nbsp;|&nbsp; [Paper](#) &nbsp;|&nbsp; [![HuggingFace](https://img.shields.io/badge/🤗%20Dataset-ELiDAR-blue)](https://huggingface.co/datasets/darshanakgr/neurolidar) &nbsp;|&nbsp; [GitHub](https://github.com/darshanakgr/neurolidar)

**Darshana Rathnayake, Dulanga Weerakoon, Meera Radhakrishnan, and Archan Misra**

---

## Abstract

LiDARs are widely used for 3D depth reconstruction, but their performance is often limited by inherent hardware constraints that impose trade-offs between range, spatial resolution, and frame rate. Many LiDAR systems typically operate at low frame rates (e.g., 5–10 Hz), prioritizing long-range sensing over responsiveness to rapid scene changes. We present **NeuroLiDAR**, an adaptive depth sensing framework that achieves effective frame rates of up to ≈66 Hz by fusing temporally sparse LiDAR data with temporally dense inputs from neuromorphic event cameras. NeuroLiDAR integrates two components: **event-based keyframe detection** and **event-guided depth extrapolation**, to dynamically adjust the sensing rate in response to scene dynamics. To evaluate our approach, we introduce **ELiDAR**, a dataset spanning outdoor and indoor scenarios, and show that NeuroLiDAR reduces depth reconstruction error by ≈29% in RMSE while achieving adaptive frame rates between 27.8–47.3 Hz.

---

## Method Overview

NeuroLiDAR operates in two stages:

1. **Keyframe Detection** — An event-based CNN (L3CNNV4) processes neuromorphic event frames to detect moments of significant scene change, triggering a new LiDAR keyframe.
2. **Depth Extrapolation** — A dual-encoder U-Net (SUNet2EnResConv) fuses sparse LiDAR depth with event voxel grids to predict dense depth maps between keyframes.

---

## Dataset: ELiDAR

The ELiDAR dataset is hosted on HuggingFace and contains paired event camera and LiDAR recordings across outdoor and indoor scenarios.

**Download:**
```
pip install huggingface_hub
```
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="darshanakgr/neurolidar", repo_type="dataset", local_dir="data/")
```

After downloading, organize the data as follows:
```
data/
├── extrapolation/
│   ├── variable_interval_voxel_grids_v1.h5
│   └── variable_interval_voxel_grids_v1_metadata.csv
└── slicing/
    └── slicing_dataset_v1.h5
```

---

## Installation

```bash
git clone https://github.com/darshanakgr/neurolidar.git
cd neurolidar
pip install torch torchvision torchmetrics h5py pandas numpy opencv-python tqdm scikit-learn tensorboard
```

---

## Training

Training hyperparameters are configured directly in each script. Edit the `config` block at the top of `main()` before running.

### Keyframe Detection

Trains **L3CNNV4** on event frames to classify keyframes. Checkpoints are saved under `checkpoints/`.

```bash
python train_keyframe_detection_model.py
```

Default hyperparameters: batch size 64, 20 epochs, lr 1e-4, NAdam optimizer, input size 640×480.

### Depth Extrapolation

Trains **SUNet2EnResConv** to predict dense depth from sparse LiDAR + event voxel grids. TensorBoard logs and checkpoints are saved under `runs/depth_extrapolation/`.

```bash
python train_extrapolation_model.py
```

Default hyperparameters: batch size 8, 50 epochs, lr 1e-3, NAdam with cosine annealing, max range 200 m.

Monitor training:
```bash
tensorboard --logdir runs/
```

---

## Evaluation

Open `evaluation.ipynb` to reproduce all reported results. The notebook covers:

- Keyframe detection: F1, precision, and recall by weather condition and data split
- Depth extrapolation: RMSE, RMSE-log, abs\_rel, sq\_rel, and δ-accuracy at multiple event window durations (4 ms, 10 ms, 20 ms, 40 ms, 100 ms)
- Comparison against the prior-depth baseline

---

## Citation

```bibtex
@inproceedings{rathnayake2026neurolidar,
  title     = {NeuroLiDAR: Adaptive Frame Rate Depth Sensing via Neuromorphic Event-LiDAR Fusion},
  author    = {Rathnayake, Darshana and Weerakoon, Dulanga and Radhakrishnan, Meera and Misra, Archan},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
