# AnonifyMe

## Description
AnonifyMe is an application that automatically detects faces in videos and blurs them, while allowing specific faces to remain unblurred using face recognition.

The system processes uploaded videos frame-by-frame using computer vision models and produces a privacy-safe output video.

## Features
- Upload a video for processing
- Automatic face detection
- Face recognition to whitelist selected identities
- Selective face blurring
- Download processed video when complete

You can add this short section to your **README.md**.

````markdown
## Setup and Run

### 1. Create a virtual environment

```bash
python -m venv .venv
````

Activate it:

Mac/Linux

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

Apply migrations:

```bash
python manage.py migrate
```

Start the server:

```bash
python manage.py runserver
```

### 4. Open the app

Visit:

```
http://127.0.0.1:8000
```