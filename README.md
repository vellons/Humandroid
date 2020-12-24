# Humandroid body pose
The goal of this project is to have a humanoid robot imitate the movements of a human.

This program use [MediaPipe](https://mediapipe.dev) to detect the human pose.<br>
MediaPipe Pose is a ML solution for high-fidelity body pose, 
inferring 33 2D landmarks on the whole body.<br>
The detector that MediaPipe uses is inspired by [BlazeFace](https://arxiv.org/abs/1907.05047).
![Dashboard](media/pose_landmarks.png)

## Project setup
#### Create a virtual environment
```shell
python3 -m venv env
```

#### Activate virtual environment
```shell
source env/bin/activate # On Linux
env\Scripts\activate # On Windows
```

#### Install all requirements inside it
```shell
pip install -r requirements.txt
```

#### Run
```shell
python3 main.py
```
