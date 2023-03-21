from typing import List

import cv2
import rospy
from utils.tello_init import initialize_tello
from utils.tracking import find_face, get_frame, track_face


class FaceTracker:
    def __init__(self) -> None:
        rospy.init_node("face_tracking", anonymous=True, log_level=rospy.DEBUG)

        self.width: float = 360
        self.height: float = 240
        self.pid: List[float] = [0.5, 0.3, 0.1]
        self.proportional_error: float = 0
        self.takeoff: bool = False

        self.tello = initialize_tello()

    def loop(self) -> None:
        while not rospy.is_shutdown():
            if not self.takeoff:
                self.tello.takeoff()
                self.takeoff = True

            img = get_frame(self.tello, self.width, self.height)

            img, info = find_face(img)

            self.proportional_error = track_face(
                self.tello, info, self.width, self.pid, self.proportional_error
            )

            cv2.imshow("Image", img)

            if cv2.waitKey(1):
                self.tello.land()
                break


if __name__ == "__main__":
    face_tracker = FaceTracker()
    face_tracker.loop()
