"""
Functions to manipulate Openpose pose estimation keypoints in JSON format.
"""
# coding: utf-8

import json
from os import listdir
import cv2



def default(o):
    return int(o)

def file_to_3d(keypoint_file, depth_image):
    """
    Take as input an existing JSON formatted openpose keypoints file, and fill the 3d keypoints
    field with information from both keypoint file and depth image.
    """
    # DEPTH_IMAGE = "buffers/buffer0/depth/depth2.png"
    # KEYPOINT_FILE = "buffers/buffer0/keypoints/color2_keypoints.json"

    with open(keypoint_file, 'r') as data_file:
        keypoints = json.load(data_file)

    depth_array = cv2.imread(depth_image, 0)


    for person in keypoints['people']:
        # Suppress joints 14 to 17
        person['pose_keypoints_2d'] = person['pose_keypoints_2d'][:3 * 14]
        
        # Fill the 3d keypoints field
        for i, elem in enumerate(person['pose_keypoints_2d']):
            if i % 3 == 2:
                person['pose_keypoints_3d'].append(depth_array[int(person['pose_keypoints_3d'][-2]),
                                                            int(person['pose_keypoints_3d'][-1])])
            person['pose_keypoints_3d'].append(elem)

    with open(keypoint_file, 'w') as outfile:
        str_ = json.dumps(keypoints, indent=4, sort_keys=False, separators=(',', ': '), ensure_ascii=False, default=default)
        outfile.write(str_)

def rep_to_3d(keypoint_rep, depth_rep):
    keypoints = listdir(keypoint_rep)
    depth_imgs = listdir(depth_rep)
    print(keypoints, depth_imgs)

    for i in range(len(keypoints)):
        file_to_3d(keypoint_rep + '/' + keypoints[i], depth_rep + '/' + depth_imgs[i])

if __name__ == "__main__":
    rep_to_3d("/home/morgan/Anaconda_projects/openpose3d/buffers/buffer0/keypoints", "/home/morgan/Anaconda_projects/openpose3d/buffers/buffer0/depth")