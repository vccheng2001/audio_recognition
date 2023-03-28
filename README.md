# Audio Recognition

## Introduction

Run apnea detection Pytorch mobile model inference through Android Studio 

## Prerequisites

* PyTorch 1.10.0 and torchaudio 0.10.0 (Optional)
* Python 3.8 (Optional)
* Android Pytorch library org.pytorch:pytorch_android_lite:1.10.0
* Android Studio 4.0.1 or later

## Quick Start


### 1. Prepare the Model

```
conda create -n pytorch_env python=3.8.5
conda activate pytorch_env
pip install torch torchaudio transformers
```

Now with PyTorch 1.10.0 and torchaudio 0.10.0 installed, run the following commands on a Terminal:

```
python create_model.py
```
This will create the PyTorch mobile interpreter model file `apnea.ptl`. Copy it to the Android app:
```

mkdir -p app/src/main/assets
cp apnea.ptl app/src/main/assets
```

### 2. Build and run with Android Studio

Start Android Studio, build and run the app on an Android device. After the app runs, tap the Start button and start recording audio; after 15 seconds, the model will run inference on the input.