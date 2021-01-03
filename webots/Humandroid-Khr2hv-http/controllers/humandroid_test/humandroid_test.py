"""humandroid_test controller."""

from controller import Robot

if __name__ == "__main__":

    # Create the Robot instance.
    robot = Robot()
    
    timestep = int(robot.getBasicTimeStep())
    max_speed = 6.28
    
    motor1 = robot.getDevice("Left_Shoulder")
    motor2 = robot.getDevice("Right_Arm1")

    
    motor1.setPosition(0)
    motor1.setVelocity(0.0)
    
    motor2.setPosition(0)
    motor2.setVelocity(0.0)
    
    # Perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:
        motor1.setPosition(6.28/360 * 90)
        motor1.setVelocity(1.0 * max_speed)
        
        motor2.setPosition(6.28/360 * -90)
        motor2.setVelocity(1.0 * max_speed)
        pass
