# Third-Party Notices

RoboLab is distributed under the [Apache License 2.0](./LICENSE).

It depends on the following third-party open-source packages, each governed by its own license. The list reflects packages declared in `pyproject.toml`; transitive dependencies carry their own upstream licenses, which take precedence over anything stated here.

## Direct dependencies

| Package | Version | License | License URL |
|---|---|---|---|
| isaaclab | 2.2.0 | BSD-3-Clause | https://github.com/isaac-sim/IsaacLab/blob/main/LICENSE |
| isaacsim | 5.0.0.0 | Apache-2.0 | https://github.com/isaac-sim/IsaacSim/blob/main/LICENSE.md |
| torch | 2.9.1+cu128 | BSD-3-Clause | https://github.com/pytorch/pytorch/blob/main/LICENSE |
| numpy | 1.26.0 | BSD-3-Clause | https://github.com/numpy/numpy/blob/main/LICENSE.txt |
| gymnasium | 1.2.0 | MIT | https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE |
| scipy | 1.15.3 | BSD-3-Clause | https://github.com/scipy/scipy/blob/main/LICENSE.txt |
| matplotlib | 3.10.0 | PSF-based (Matplotlib License) | https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE |
| h5py | 3.16.0 | BSD-3-Clause | https://github.com/h5py/h5py/blob/master/LICENSE |
| PyYAML | 6.0.2 | MIT | https://github.com/yaml/pyyaml/blob/main/LICENSE |
| pandas | 3.0.2 | BSD-3-Clause | https://github.com/pandas-dev/pandas/blob/main/LICENSE |
| pyarrow | 24.0.0 | Apache-2.0 | https://github.com/apache/arrow/blob/main/LICENSE.txt |
| huggingface_hub | 0.35.3 | Apache-2.0 | https://github.com/huggingface/huggingface_hub/blob/main/LICENSE |
| psutil | 5.9.8 | BSD-3-Clause | https://github.com/giampaolo/psutil/blob/master/LICENSE |
| setuptools | 80.10.2 | MIT | https://github.com/pypa/setuptools/blob/main/LICENSE |

## System dependencies

The following are not Python packages and must be installed via the host OS package manager:

| Component | Source | Notes |
|---|---|---|
| ffmpeg | distro package (`apt install ffmpeg`, etc.) | Used for video encoding during dataset export. License terms are governed by the distribution's ffmpeg build. |

## Bundled assets

Only the assets under `assets/` listed below are included. Where a folder ships its own `LICENSE` file, that local license governs and takes precedence over the entry below.

| Path | License | Source / Provenance |
|---|---|---|
| `assets/objects/ycb` | MIT | Derivative of the [YCB-Video](https://github.com/yuxng/YCB_Video_toolbox) dataset, © 2017 Yu Xiang. See [`assets/objects/ycb/LICENSE`](./assets/objects/ycb/LICENSE). |
| `assets/fixtures` | CC BY-NC-SA 4.0 | © 2026 NVIDIA Corporation. |
| `assets/robots/fr3_official` | Apache-2.0 | Converted from [franka_description](https://github.com/frankarobotics/franka_description), © Franka Robotics GmbH. See [`LICENSE`](./assets/robots/fr3_official/LICENSE), [`NOTICE`](./assets/robots/fr3_official/NOTICE) and [`PROVENANCE.md`](./assets/robots/fr3_official/PROVENANCE.md) in that folder. |
