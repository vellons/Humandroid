# Humandroid body pose

The goal of this project is to have a REAL humanoid robot imitate the movements of a human.

The example below shows a simulation with Webots.
![Webots simulation](media/webots_simulation.gif)

This program use [MediaPipe](https://mediapipe.dev) to detect the body pose.<br>
MediaPipe Pose is a ML solution for high-fidelity body pose, 
inferring 33 2D landmarks on the whole body.<br>
The detector that MediaPipe uses is inspired by [BlazeFace](https://arxiv.org/abs/1907.05047).

![Body pose landmarks](media/pose_landmarks.png)

## How it works

This program contains a Humandroid class, which takes care of all image processing operations, using MediaPipe.
The Humandroid class contains various methods that perform operations/calculations on the image. 
The results are always accessible from the main program and can be exported, in JSON format, 
to communicate with other systems and control a humanoid robot.

In the example below, the yellow labels represent the angles that the servomotors of a humanoid robot must assume.

![Body pose example](media/pose_full_body_example.gif)

JSON data representation example: (only 1 landmark)
```
{
    "pose_landmarks": [
        {
            "id": 13,
            "name": "PoseLandmark.LEFT_ELBOW",
            "x": 0.7578693628311157,
            "y": 1.1605275869369507,
            "z": -0.3892576992511749,
            "visibility": 0.8600940108299255,
            "angle": 52,
            "z_angle": 5
        }
    ]
}
```

## 3D environment

Example of a "3d environment" with matplotlib, build in realtime.

![Body 3d environment](media/pose_3d_environment.gif)


## Project setup

#### Create a virtual environment
```
python3 -m venv env
```

#### Activate virtual environment
```
source env/bin/activate # On Linux
env\Scripts\activate # On Windows
```

#### Install all requirements inside it
```
pip install -r requirements.txt
```

#### Run
```shell
python3 main.py
```
