"""
humandroid_websocket_client controller.
Required: https://github.com/vellons/SimplePYBotSDK
"""
import asyncio
import websockets
import json
from controller import Robot

SOCKET_HOST = "localhost"
SOCKET_PORT = 65432
PATH = "/"
MAX_SPEED = 6.28  # Max motor speed

# Create the Robot instance
robot = Robot()


def move_robot(motors):
    for m in motors:
        motor = robot.getDevice(m["id"])
        motor.setPosition(6.28 / 360 * m["abs_current_angle"])
        motor.setVelocity(1 * MAX_SPEED)


async def loop():
    websocket = await websockets.connect("ws://" + SOCKET_HOST + ":" + str(SOCKET_PORT) + PATH)
    await websocket.send('{"socket": {"format": "absolute"}}'.encode())

    while robot.step(time_step) != -1:
        try:
            data = await websocket.recv()
            data = json.loads(data)
            print(data)
            move_robot(data["motors"])
        except:
            pass


if __name__ == "__main__":
    time_step = int(robot.getBasicTimeStep())

    asyncio.get_event_loop().run_until_complete(loop())
