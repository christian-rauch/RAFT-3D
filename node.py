#! /usr/bin/env python3

import torch
import torch.nn.functional as F
# from raft3d.raft3d import RAFT3D
import raft3d.projective_ops as pops
# from scripts.demo import prepare_images_and_depths
from argparse import Namespace
from scripts.utils import show_image, normalize_image
import numpy as np
# import matplotlib.pyplot as plt
import importlib

import rospy
import message_filters
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
import collections
import cv2
from cv_bridge import CvBridge

DEPTH_SCALE = 0.2

def prepare_images_and_depths(image1, image2, depth1, depth2):
    image1 = F.pad(image1, [0,0,0,4], mode='replicate')
    image2 = F.pad(image2, [0,0,0,4], mode='replicate')
    depth1 = F.pad(depth1[:,None], [0,0,0,4], mode='replicate')[:,0]
    depth2 = F.pad(depth2[:,None], [0,0,0,4], mode='replicate')[:,0]

    depth1 = (DEPTH_SCALE * depth1).float()
    depth2 = (DEPTH_SCALE * depth2).float()
    image1 = normalize_image(image1)
    image2 = normalize_image(image2)

    return image1, image2, depth1, depth2


class Node:
    def __init__(self):
        # network
        model_args = Namespace(network = "raft3d.raft3d", model = "raft3d.pth")
        # model_args = Namespace(network = "raft3d.raft3d_bilaplacian", model = "raft3d_kitti.pth")
        RAFT3D = importlib.import_module(model_args.network).RAFT3D
        self.model = torch.nn.DataParallel(RAFT3D(model_args))
        self.model.load_state_dict(torch.load(model_args.model), strict=False)

        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

        # ROS
        rospy.init_node('raft3d')
        self.bridge = CvBridge()

        self.imgs = collections.deque(maxlen=2)

        sub_ci = message_filters.Subscriber("/rgb/camera_info", CameraInfo)
        sub_colour = message_filters.Subscriber("/rgb/image_raw/compressed", CompressedImage)
        sub_depth = message_filters.Subscriber("/depth_to_rgb/image_raw/filtered/compressed", CompressedImage)

        ts = message_filters.ApproximateTimeSynchronizer([sub_ci, sub_colour, sub_depth], 10, 0.1, allow_headerless=False)
        ts.registerCallback(self.on_images)

        rospy.Timer(rospy.Duration(0.1), self.process, reset=True)

        print("ready")

    def run(self):
        try:
            rospy.spin()
        except rospy.exceptions.ROSTimeMovedBackwardsException:
            pass
    
    def on_images(self, msg_info, msg_colour, msg_depth):
        # print("sync")
        colour = self.bridge.compressed_imgmsg_to_cv2(msg_colour, desired_encoding='passthrough')
        depth = self.bridge.compressed_imgmsg_to_cv2(msg_depth, desired_encoding='passthrough')

        # convert to metre
        # depth = depth / 1000.0
        depth = depth.astype(float)

        # scale = 1 # out of memory
        # scale = 540/720
        scale = 3/4 # scale 1280x720 to 960x540

        colour = cv2.resize(colour, None, fx=scale, fy=scale)
        depth = cv2.resize(depth, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        #     [fx'  0  cx' Tx]
        # P = [ 0  fy' cy' Ty]
        #     [ 0   0   1   0]
        fx = msg_info.P[0] * scale
        fy = msg_info.P[5] * scale
        cx = msg_info.P[2] * scale
        cy = msg_info.P[6] * scale

        # print(colour.shape, colour.dtype, depth.shape, depth.dtype)

        # cv2.imshow("colour", colour)
        # cv2.imshow('depth', depth/2.0)
        # cv2.waitKey(1)

        self.imgs.append(([fx, fy, cx, cy], colour, depth))

    @torch.no_grad()
    def predict(self, ci, img1, img2, depth1, depth2):
        print("predict")
        cv2.imshow("i1", img1)
        cv2.imshow("i2", img2)
        cv2.imshow("d1", depth1)
        cv2.imshow("d2", depth2)

        print(depth1.shape, depth1.dtype, np.min(depth1), np.max(depth1))

        # depth1 = torch.from_numpy(fx / disp1).float().cuda().unsqueeze(0)
        # depth2 = torch.from_numpy(fx / disp2).float().cuda().unsqueeze(0)
        depth1 = torch.from_numpy(depth1).float().cuda().unsqueeze(0)
        depth2 = torch.from_numpy(depth2).float().cuda().unsqueeze(0)
        image1 = torch.from_numpy(img1).permute(2,0,1).float().cuda().unsqueeze(0)
        image2 = torch.from_numpy(img2).permute(2,0,1).float().cuda().unsqueeze(0)
        intrinsics = torch.as_tensor(ci).cuda().unsqueeze(0)

        print(intrinsics.shape, image1.shape, image2.shape, depth1.shape, depth2.shape)
        image1, image2, depth1, depth2 = prepare_images_and_depths(image1, image2, depth1, depth2)
        Ts = self.model(image1, image2, depth1, depth2, intrinsics, iters=16)
    
        # compute 2d and 3d from from SE3 field (Ts)
        flow2d, flow3d, _ = pops.induced_flow(Ts, depth1, intrinsics)

        # extract rotational and translational components of Ts
        tau, phi = Ts.log().split([3,3], dim=-1)
        tau = tau[0].cpu().numpy()
        phi = phi[0].cpu().numpy()

        # undo depth scaling
        flow3d = flow3d / DEPTH_SCALE

        self.display(img1, tau, phi)

    def display(self, img, tau, phi):
        print("display")
        # fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        # ax1.imshow(img[:, :, ::-1] / 255.0)
        cv2.imshow("colour", img[:, :, ::-1] / 255.0)

        tau_img = np.clip(tau, -0.1, 0.1)
        tau_img = (tau_img + 0.1) / 0.2

        phi_img = np.clip(phi, -0.1, 0.1)
        phi_img = (phi_img + 0.1) / 0.2

        # print(tau_img)

        print(tau_img.shape, tau_img.dtype, np.min(tau_img),np.max(tau_img))

        # ax2.imshow(tau_img)
        cv2.imshow("tau", tau_img)
        # ax3.imshow(phi_img)
        cv2.imshow("phi", phi_img)
        # plt.show()
        cv2.waitKey(1)

    def process(self, event):
        if len(self.imgs)<2:
            return

        # print("process")

        ci = self.imgs[0][0]
        image1 = self.imgs[0][1]
        depth1 = self.imgs[0][2]
        image2 = self.imgs[1][1]
        depth2 = self.imgs[1][2]

        self.predict(ci, image1, image2, depth1, depth2)

        # remove the oldest image so we do not process the same pair again
        self.imgs.popleft()


if __name__ == '__main__':
    try:
        Node().run()
    except rospy.exceptions.ROSTimeMovedBackwardsException:
        pass
