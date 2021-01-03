"""humandroid_up_down controller."""

from controller import Robot
from time import sleep

if __name__ == "__main__":

    # Create the Robot instance.
    robot = Robot()
    
    timestep = int(robot.getBasicTimeStep())
    max_speed = 6.28
    vel = 1.0 * max_speed
    
    motor10 = robot.getDevice("Left_Leg1")
    motor11 = robot.getDevice("Left_Leg3")
    motor12 = robot.getDevice("Left_Ankle")
    motor20 = robot.getDevice("Right_Leg1")
    motor21 = robot.getDevice("Right_leg3")
    motor22 = robot.getDevice("Right_Ankle")


    motor10.setPosition(0)
    motor11.setPosition(0)
    motor12.setPosition(0)
    motor20.setPosition(0)
    motor21.setPosition(0)
    motor22.setPosition(0)

    
    motor10.setVelocity(vel)
    motor11.setVelocity(vel)
    motor12.setVelocity(vel)
    motor20.setVelocity(vel)
    motor21.setVelocity(vel)
    motor22.setVelocity(vel)
    
    # Perform simulation steps until Webots is stopping the controller
    angle = 0
    sum = 1  
    while robot.step(timestep) != -1:
        angle += sum
        
        if angle >= 77:
            sum = -sum
        elif angle <= 0:
            sum = -sum
        print(angle)
 
        motor10.setPosition(6.28/360 * angle)
        motor11.setPosition(6.28/360 * -angle*2)
        motor12.setPosition(6.28/360 * -angle)
        motor20.setPosition(6.28/360 * -angle)
        motor21.setPosition(6.28/360 * angle*2)
        motor22.setPosition(6.28/360 * angle)
        pass
