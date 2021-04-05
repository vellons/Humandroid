"""
Use this controller with Webots. Works with khr-2hv robot model
Required: https://github.com/vellons/SimplePYBotSDK
"""
import socket
import json
from controller import Robot

SOCKET_HOST = "localhost"
SOCKET_PORT = 65432
MAX_SPEED = 6.28  # Max motor speed

# Create the Robot instance
robot = Robot()


def move_robot(motors):
    for m in motors:
        motor = robot.getDevice(m["id"])
        motor.setPosition(6.28 / 360 * m["abs_current_angle"])
        motor.setVelocity(1 * MAX_SPEED)


if __name__ == "__main__":
    time_step = int(robot.getBasicTimeStep())

    socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket.connect((SOCKET_HOST, SOCKET_PORT))
    socket.settimeout(0.003)
    socket.send('{"socket": {"format": "absolute"}}'.encode())

    while robot.step(time_step) != -1:
        try:
            data = socket.recv(8192).decode()
            data = json.loads(data)
            print(data)
            move_robot(data["motors"])
        except:
            pass
