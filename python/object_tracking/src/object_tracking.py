from typing import List

import cv2
import numpy as np
import rospy
from utils.tello_init import initialize_tello
from utils.tracking import get_contours, stack_images


class FaceTracker:
    def __init__(self) -> None:
        rospy.init_node("object_tracking", anonymous=True, log_level=rospy.DEBUG)

        self.width: float = 360
        self.height: float = 240
        self.frame_width = self.width
        self.frame_height = self.height
        self.dead_zone: float = 100
        self.pid: List[float] = [0.5, 0.3, 0.1]
        self.proportional_error: float = 0
        self.takeoff: bool = False

        self.tello = initialize_tello()

    def display(self, img: cv2.Mat) -> None:
        cv2.line(
            img,
            (int(self.frame_width / 2) - self.dead_zone, 0),
            (int(self.frame_width / 2) - self.dead_zone, self.frame_height),
            (255, 255, 0),
            3,
        )
        cv2.line(
            img,
            (int(self.frame_width / 2) + self.dead_zone, 0),
            (int(self.frame_width / 2) + self.dead_zone, self.frame_height),
            (255, 255, 0),
            3,
        )
        cv2.circle(img, (int(self.frame_width / 2), int(self.frame_height / 2)), 5, (0, 0, 255), 5)
        cv2.line(
            img,
            (0, int(self.frame_height / 2) - self.dead_zone),
            (self.frame_width, int(self.frame_height / 2) - self.dead_zone),
            (255, 255, 0),
            3,
        )
        cv2.line(
            img,
            (0, int(self.frame_height / 2) + self.dead_zone),
            (self.frame_width, int(self.frame_height / 2) + self.dead_zone),
            (255, 255, 0),
            3,
        )

    def loop(self) -> None:
        while not rospy.is_shutdown():
            if not self.takeoff:
                self.tello.takeoff()
                self.takeoff = True

            frame_read = self.tello.get_frame_read()
            frame = frame_read.frame
            img = cv2.resize(frame, (self.width, self.height))
            img_contour = img.copy()
            imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            h_min = cv2.getTrackbarPos("HUE Min", "HSV")
            h_max = cv2.getTrackbarPos("HUE Max", "HSV")
            s_min = cv2.getTrackbarPos("SAT Min", "HSV")
            s_max = cv2.getTrackbarPos("SAT Max", "HSV")
            v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
            v_max = cv2.getTrackbarPos("VALUE Max", "HSV")

            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            mask = cv2.inRange(imgHsv, lower, upper)
            result = cv2.bitwise_and(img, img, mask=mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            imgBlur = cv2.GaussianBlur(result, (7, 7), 1)
            imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
            threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
            threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
            imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
            kernel = np.ones((5, 5))
            imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
            direction = get_contours(
                imgDil, img_contour, self.frame_width, self.frame_height, self.dead_zone
            )
            self.display(img_contour)

            if direction == 1:
                self.tello.yaw_velocity = -60
            elif direction == 2:
                self.tello.yaw_velocity = 60
            elif direction == 3:
                self.tello.up_down_velocity = 60
            elif direction == 4:
                self.tello.up_down_velocity = -60
            else:
                self.tello.left_right_velocity = 0
                self.tello.for_back_velocity = 0
                self.tello.up_down_velocity = 0
                self.tello.yaw_velocity = 0
            # SEND VELOCITY VALUES TO TELLO
            if self.tello.send_rc_control:
                self.tello.send_rc_control(
                    self.tello.left_right_velocity,
                    self.tello.for_back_velocity,
                    self.tello.up_down_velocity,
                    self.tello.yaw_velocity,
                )
                rospy.logdebug(direction)

            stack = stack_images(0.9, ([img, result], [imgDil, img_contour]))
            cv2.imshow("Horizontal Stacking", stack)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.tello.land()
                break


if __name__ == "__main__":
    face_tracker = FaceTracker()
    face_tracker.loop()
