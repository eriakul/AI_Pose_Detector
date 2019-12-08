import os
from glob import iglob
import cv2
import numpy as np
from math import sin, cos, radians
from functools import partial
 

class Mother:
    def __init__(self):
        self.body_angles = {
            "left_shoulder": 0,
            "left_elbow": 0,
            "right_shoulder": 0,
            "right_elbow": 0,
            "left_hip": 0,
            "left_knee": 0,
            "right_hip": 0,
            "right_knee": 0
        }
        self.body_locations = {
          "body": [(310, 210), (390, 310)],
          "left_shoulder_joint": (310, 210),
          "right_shoulder_joint": (390, 210),
          "left_hip_joint": (310, 310),
          "right_hip_joint": (390, 310)
        }

        self.body_lengths = {
            "biceps": 100,
            "forearms": 100,
            "thighs": 100,
            "calves": 100
        }
    
    def rotate(self, joint, x):
        print(x, joint)
        self.body_angles[joint] = radians(x)
    
    def get_endpoint(self, start, angle, length, zero):
        if zero == "up":
            return (int(start[0] + length*sin(angle)), int(start[1] + length*cos(angle)))
        if zero == "down":
            pass

    def draw_lollipop(self, start, end, canvas):



    def render_body(self):       
        # initialize canvas
        target_size = 700
        canvas = np.full((target_size, target_size, 3), 255, np.uint8)
        # draw body 
        body1, body2 = self.body_locations["body"]
        canvas = cv2.rectangle(canvas, body1, body2, (100, 100, 100), -1)
        # draw right arm
        start, length, angle = self.body_locations["right_shoulder_joint"], self.body_lengths["biceps"], self.body_angles["right_shoulder"]
        endpoint = self.get_endpoint(start, angle, length,"up" )
        canvas = cv2.circle(canvas, endpoint, 10, (0, 0, 255), -1) 
        return canvas



mother = Mother()

rootdir_glob = 'C:\\Users\\elu\\Documents\\_Code\Machine Learning Final\\AI_Pose_Detector\\data' # Note the added asterisks
file_list = [f for f in iglob('*/*', recursive=True) if os.path.isfile(f)]
f = file_list[0]
cv2.namedWindow("Sketch")

trackbars = [("right_shoulder", 180, 90), ("left_shoulder", 180, 90)]
for name, maxVal, defaultVal in trackbars:
    cv2.createTrackbar(name,"Sketch", 0, maxVal, partial(mother.rotate, name))
    cv2.setTrackbarPos(name, "Sketch", defaultVal)

while True:
    image = mother.render_body()
    cv2.imshow("Sketch", image)
    key = cv2.waitKey(1)
