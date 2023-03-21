import rospy
from djitellopy import Tello


def initialize_tello() -> Tello:
    tello = Tello()
    tello.connect()
    tello.for_back_velocity = 0
    tello.left_right_velocity = 0
    tello.up_down_velocity = 0
    tello.yaw_velocity = 0
    tello.speed = 0
    rospy.logdebug(tello.get_battery())
    tello.streamoff()
    tello.streamon()
    return tello
