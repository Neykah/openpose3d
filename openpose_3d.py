"""
Main thread: Using libfreenect2 multiframe listener, store N RGB and depth frames in a buffer repository.
When it is full, call a slave thread giving it the frame number as input, and continue in another buffer repository.

Slave thread: Call openpose on the RGB images in the current repository and save the keypoints in an output folder.
Then, modify these keypoints by adding the depth information. When done, remove the buffer repository.

Author: Morgan Lefranc
Date: 01/06/2018
"""

# coding: utf-8
# pylint: disable-msg=C0103,E0611, C0411, I1101

import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel
import subprocess
import threading
import os

import json_manip

# Parameters
N = 5
bufferIndex = 0
bufferFrameIndex = 0
cwd = os.getcwd()

class openPoseThread(threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID
    
    def run(self):
        global cwd
        print("Starting thread nb", self.threadID)
        buffer_dir = cwd + "/buffers/buffer{}".format(self.threadID)
        color_dir = buffer_dir + "/color"
        depth_dir = buffer_dir + "/depth"
        keypoints_dir = buffer_dir + "/keypoints"
        # YAML save version (deprecated)
        # subprocess.call("./build/examples/openpose/openpose.bin --image_dir {} --write_images {} --write_keypoint {} --write_keypoint_format yml --display 0".format(color_dir, color_dir, keypoints_dir), shell=True)

        # JSON save version
        subprocess.call("./build/examples/openpose/openpose.bin --image_dir {} --write_images {} --write_keypoint_json {} --display 0".format(color_dir, color_dir, keypoints_dir), shell=True)
        json_manip.rep_to_3d(keypoints_dir, depth_dir)
        print("Ending thread nb", self.threadID)



def createBuffer():
    global bufferIndex, cwd
    subprocess.call("mkdir {}/buffers/buffer{}".format(cwd, bufferIndex), shell=True)
    subprocess.call("mkdir {}/buffers/buffer{}/color".format(cwd, bufferIndex), shell=True)
    subprocess.call("mkdir {}/buffers/buffer{}/depth".format(cwd, bufferIndex), shell=True)
    subprocess.call("mkdir {}/buffers/buffer{}/keypoints".format(cwd, bufferIndex), shell=True)
    bufferIndex += 1

def deleteBuffer(i):
    global cwd
    subprocess.call("rm -rf {}/buffers/buffer{}".format(cwd, i), shell=True)


# Remove all existing buffer repositories
subprocess.call("rm -rf {}/buffers/buffer*".format(cwd), shell=True)

# Create first buffer
createBuffer()

# Move to openpose root
os.chdir("/home/morgan/openpose")

try:
    from pylibfreenect2 import OpenCLPacketPipeline
    pipeline = OpenCLPacketPipeline()

except:
    try:
        from pylibfreenect2 import OpenGLPacketPipeline
        pipeline = OpenGLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()
print("Packet pipeline: ", type(pipeline).__name__)

# Create and set logger
logger = createConsoleLogger(LoggerLevel.Debug)
setGlobalLogger(logger)

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No devices connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

listener = SyncMultiFrameListener(
    FrameType.Color | FrameType.Ir | FrameType.Depth)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

device.start()

# NOTE: must be called after device.start()
registration = Registration(device.getIrCameraParams(),
                            device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)

# Optional parameters for registration
# set True if needed
need_bigdepth = False
need_color_depth_map = False

bigdepth = Frame(1920, 1082, 4) if need_bigdepth else None
color_depth_map = np.zeros((424, 512), np.int32).ravel() \
    if need_color_depth_map else None

while True:
    frames = listener.waitForNewFrame()

    color = frames["color"]
    ir = frames["ir"]
    depth = frames["depth"]

    registration.apply(color, depth, undistorted, registered, bigdepth=bigdepth, color_depth_map=color_depth_map)

    # cv2.imshow("ir", ir.asarray() / 65535.)
    depth_array = depth.asarray()
    # cv2.imshow("depth", depth_array)
    color_array = cv2.resize(color.asarray(), (int(1920 / 3), int(1080 / 3)))
    cv2.imshow("color", color_array)
    
    reg_array = registered.asarray(np.uint8)
    cv2.imshow("registered", reg_array)

    # if need_bigdepth:
    #     cv2.imshow("bigdepth", cv2.resize(bigdepth.asarray(np.float32),
    #                                       (int(1920 / 3), int(1082 / 3))))
    
    # Tentative aborted: I choose to use "registered" in the end.
    # if need_color_depth_map:
    #     img = color_depth_map
    #     print("img1", img.shape)
    #     img2 = (img + 2**16).astype(np.uint32)
    #     print("img2", img2.shape)
    #     img3 = img2.view(np.uint8).reshape(img2.shape + (4,))[:, :3]
    #     print("img3", img3.shape)
    #     img4 = img3.reshape(424, 512, 3)
    #     cv2.imshow("color_depth_map", img.reshape(424, 512))
    #     cv2.imshow("new_color_map", img4)

    if bufferFrameIndex < N:
        # Option to write raw color images
        # cv2.imwrite("{}/buffers/buffer{}/color/true_color{}.png".format(cwd, bufferIndex - 1, bufferFrameIndex), color_array)
        cv2.imwrite("{}/buffers/buffer{}/color/color{}.png".format(cwd, bufferIndex - 1, bufferFrameIndex), reg_array)
        cv2.imwrite("{}/buffers/buffer{}/depth/depth{}.png".format(cwd, bufferIndex - 1, bufferFrameIndex), depth_array / 10.)
        bufferFrameIndex += 1
    
    else:
        bufferFrameIndex = 0
        new_thread = openPoseThread(bufferIndex - 1)
        new_thread.start()
        createBuffer()
        


    listener.release(frames)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

device.stop()
device.close()

sys.exit(0)
