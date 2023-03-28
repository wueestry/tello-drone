from typing import Tuple

import cv2
import numpy as np
from djitellopy import BackgroundFrameRead, Tello


def stack_images(scale: float, img_array: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(
                        img_array[x][y],
                        (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                        None,
                        scale,
                        scale,
                    )
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(
                    img_array[x],
                    (img_array[0].shape[1], img_array[0].shape[0]),
                    None,
                    scale,
                    scale,
                )
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


def get_contours(
    img: cv2.Mat, img_contour: cv2.Mat, frame_width: float, frame_height: float, dead_zone: float
) -> float:
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            cv2.drawContours(img_contour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print(len(approx))
            x, y, w, h = cv2.boundingRect(approx)
            cx = int(x + (w / 2))  # CENTER X OF THE OBJECT
            cy = int(y + (h / 2))  # CENTER X OF THE OBJECT

            if cx < int(frame_width / 2) - dead_zone:
                cv2.putText(
                    img_contour, " GO LEFT ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3
                )
                cv2.rectangle(
                    img_contour,
                    (0, int(frame_height / 2 - dead_zone)),
                    (int(frame_width / 2) - dead_zone, int(frame_height / 2) + dead_zone),
                    (0, 0, 255),
                    cv2.FILLED,
                )
                direction = 1
            elif cx > int(frame_width / 2) + dead_zone:
                cv2.putText(
                    img_contour,
                    " GO RIGHT ",
                    (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )
                cv2.rectangle(
                    img_contour,
                    (int(frame_width / 2 + dead_zone), int(frame_height / 2 - dead_zone)),
                    (frame_width, int(frame_height / 2) + dead_zone),
                    (0, 0, 255),
                    cv2.FILLED,
                )
                direction = 2
            elif cy < int(frame_height / 2) - dead_zone:
                cv2.putText(
                    img_contour, " GO UP ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3
                )
                cv2.rectangle(
                    img_contour,
                    (int(frame_width / 2 - dead_zone), 0),
                    (int(frame_width / 2 + dead_zone), int(frame_height / 2) - dead_zone),
                    (0, 0, 255),
                    cv2.FILLED,
                )
                direction = 3
            elif cy > int(frame_height / 2) + dead_zone:
                cv2.putText(
                    img_contour, " GO DOWN ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3
                )
                cv2.rectangle(
                    img_contour,
                    (int(frame_width / 2 - dead_zone), int(frame_height / 2) + dead_zone),
                    (int(frame_width / 2 + dead_zone), frame_height),
                    (0, 0, 255),
                    cv2.FILLED,
                )
                direction = 4
            else:
                direction = 0

            cv2.line(
                img_contour,
                (int(frame_width / 2), int(frame_height / 2)),
                (cx, cy),
                (0, 0, 255),
                3,
            )
            cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 5)
            cv2.putText(
                img_contour,
                "Points: " + str(len(approx)),
                (x + w + 20, y + 20),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                img_contour,
                "Area: " + str(int(area)),
                (x + w + 20, y + 45),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                img_contour,
                " " + str(int(x)) + " " + str(int(y)),
                (x - 20, y - 45),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            direction = 0
        return direction
