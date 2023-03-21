from typing import List, Tuple

import cv2
import numpy as np
import rospy
from djitellopy import BackgroundFrameRead, Tello


def get_frame(tello: Tello, width: float = 360, height: float = 240) -> BackgroundFrameRead:
    frame = tello.get_frame_read()
    frame = frame.frame
    img = cv2.resize(frame, (width, height))
    return img


def find_face(img: BackgroundFrameRead) -> Tuple[BackgroundFrameRead, list]:
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, 1.1, 6)

    face_list_c = []
    face_list_area = []

    for x, y, width, height in faces:
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 2)
        cx = x + width // 2
        cy = y + height // 2
        area = width * height
        face_list_area.append(area)
        face_list_c.append([cx, cy])

    if len(face_list_area) != 0:
        i = face_list_area.index(max(face_list_area))
        return img, [face_list_c[i], face_list_area[i]]
    else:
        return img, [[0, 0], 0]


def track_face(
    tello: Tello,
    info: List[list],
    width: float,
    pid: List[float],
    proportional_error: float,
) -> float:
    ## PID
    error = info[0][0] - width // 2
    speed = pid[0] * error + pid[1] * (error - proportional_error)
    speed = int(np.clip(speed, -100, 100))

    rospy.logdebug(speed)
    if info[0][0] != 0:
        tello.yaw_velocity = speed
    else:
        tello.for_back_velocity = 0
        tello.left_right_velocity = 0
        tello.up_down_velocity = 0
        tello.yaw_velocity = 0
        error = 0
    if tello.send_rc_control:
        tello.send_rc_control(
            tello.left_right_velocity,
            tello.for_back_velocity,
            tello.up_down_velocity,
            tello.yaw_velocity,
        )
    return error
