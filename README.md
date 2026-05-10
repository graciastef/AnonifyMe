# AnonifyMe

AnonifyMe automatically detects faces in uploaded videos, blurs them, and lets selected whitelisted faces remain visible.

## Features

- Upload videos for processing
- Detect and track faces frame by frame
- Whitelist selected identities with face recognition
- Blur non-whitelisted faces
- Download processed videos
- Capture low-confidence detections for SCRFD retraining

## Environments

Use conda for local setup. The web app and SCRFD retraining use separate environments because they need different Python and ML dependency stacks.

## Web App

Create and activate the app environment:

```bash
conda create -y -n anonifyme-app python=3.12
conda activate anonifyme-app
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file with the Azure and ClearML values used by `config/settings.py`, then run:

```bash
python manage.py migrate
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000
```

For the Django shell:

```bash
python manage.py shell
```

## SCRFD Retraining

SCRFD retraining uses the older InsightFace/MMDetection stack, so run it in its own conda environment:

```bash
conda create -y -n anonifyme-retraining python=3.9
conda activate anonifyme-retraining

python -m pip install --upgrade pip "setuptools<70" wheel "cython==0.29.33"
conda install -y "mkl<2024" numpy=1.23.5
conda install -y -c pytorch pytorch==1.10.2 torchvision==0.11.3 cudatoolkit=11.3

python -m pip install mmcv-full==1.3.18 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

python -m pip install --no-build-isolation -r retraining/requirements.txt
```

If `mmcv-full` fails through plain pip, install it after PyTorch with build isolation disabled:

```bash
pip install --no-build-isolation mmcv-full==1.3.18
pip install --no-build-isolation -r retraining/requirements.txt
```

Retraining uses OpenMMLab's older `mmpycocotools` package. Keep `cython==0.29.33` installed before installing `retraining/requirements.txt`, and install the requirements with `--no-build-isolation` so the legacy build uses the conda env's Cython/Numpy stack.

Run retraining:

```bash
python -m retraining
```

If installing retraining dependencies fails with:

```text
AttributeError: module 'pkgutil' has no attribute 'ImpImporter'
ERROR: Failed to build 'numpy'
```

you are installing the retraining requirements in the wrong Python environment. Check:

```bash
python --version
```

Retraining must run in the `anonifyme-retraining` conda environment with Python 3.9:

```bash
conda activate anonifyme-retraining
python --version
pip install --no-build-isolation -r retraining/requirements.txt
```

Retraining expects:

- ClearML environment variables in `.env` or an existing ClearML config
- Azure Blob credentials in `.env`
- WiderFace/SCRFD labels under `retraining/dataset/`
- The pretrained SCRFD checkpoint uploaded to Azure at `models/scrfd/pretrained/scrfd_10g.pth`

Retraining runs are logged in the ClearML project:

```text
anonifyme-scrfd-retrain
```

Video-processing runs are logged in:

```text
anonifyme-video-processing
```
