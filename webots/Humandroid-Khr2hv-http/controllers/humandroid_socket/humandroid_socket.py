"""
humandroid_socket controller.
Required: https://github.com/vellons/SimplePYBotSDK
"""

from controller import Robot
import socket
import json

HOST = "127.0.0.1"
PORT = 65432
MAX_SPEED = 6.28  # Max motor speed

# Create the Robot instance
robot = Robot()


def move_robot(motors):
    motors = motors.decode()
    motors = json.loads(motors)
    print(motors)
    
    for m in motors:
        motor = robot.getDevice(m["id"])
        motor.setPosition(6.28/360 * m["angle"])
        motor.setVelocity(1 * MAX_SPEED)
        

if __name__ == "__main__":

    timestep = int(robot.getBasicTimeStep())
    
    # Create socket with 1 listener
    socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket.bind((HOST, PORT))
    socket.listen(1)
    
    # Wait for connection
    conn, addr = socket.accept()
    conn.settimeout(0.003)
    print("Connected from: {}".format(addr))
    
    with conn:
        # Perform simulation steps until Webots is stopping the controller
        print("Start robot step simulation")
        while robot.step(timestep) != -1:
            
            try:
                data = conn.recv(8192)
                move_robot(data)
            except Exception as e:
                if str(e) == "timed out":
                    # If timeout return control to webots
                    pass
                else:
                    # Print error and stop
                    print(e)
                    pass
            
            pass
