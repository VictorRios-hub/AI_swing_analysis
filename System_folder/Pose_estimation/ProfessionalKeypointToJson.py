import cv2
import mediapipe as mp
import os
import numpy as np
import json
#This system writes the professional keypoints to JSON file
mpPose = mp.solutions.pose
pose = mpPose.Pose(model_complexity=2, smooth_landmarks=False, static_image_mode=True)

keypoints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

#file directories
directory = '../golfdb/ImageData/Pro_Swing_events'
json_directory = 'Keypoint_data/ProfessionalKeypointData'

#moment of swing list
moments_of_swing = ['Address', 'Mid-Backswing (arm parallel)', 'Mid-Downswing (arm parallel)', 'Mid-Follow-Through (shaft parallel)', 'Top', 'Toe-up', 'Impact', 'Finish']

#loop through each moment of swing file
for moment in os.listdir(directory):
    moment_dir = os.path.join(directory, moment)
    moment_json_dir = os.path.join(json_directory, moment)

    if not os.path.exists(moment_json_dir):
        os.makedirs(moment_json_dir)

    #loop through each image
    for filename in os.listdir(moment_dir):
        #read file and calculate keypoint data
        f = os.path.join(moment_dir, filename)
        img = cv2.imread(f)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        if results.pose_landmarks:
            keypoint_data = {}
            for id, lm in enumerate(results.pose_landmarks.landmark):
                if id in keypoints:
                    keypoint_data[id] = [lm.x, lm.y, abs(lm.z)]

        #write keypoint data to file
        json_filename = os.path.splitext(filename)[0] + '.json'
        with open(os.path.join(moment_json_dir, json_filename), 'w') as file:
            json.dump(keypoint_data, file, indent=4)
