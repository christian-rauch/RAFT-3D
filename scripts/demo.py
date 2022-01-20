#!/usr/bin/env python3
import sys
sys.path.append('.')

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from lietorch import SE3
import raft3d.projective_ops as pops
from data_readers import frame_utils
from utils import show_image, normalize_image


DEPTH_SCALE = 0.2

def prepare_images_and_depths(image1, image2, depth1, depth2):
    """ padding, normalization, and scaling """

    image1 = F.pad(image1, [0,0,0,4], mode='replicate')
    image2 = F.pad(image2, [0,0,0,4], mode='replicate')
    depth1 = F.pad(depth1[:,None], [0,0,0,4], mode='replicate')[:,0]
    depth2 = F.pad(depth2[:,None], [0,0,0,4], mode='replicate')[:,0]

    depth1 = (DEPTH_SCALE * depth1).float()
    depth2 = (DEPTH_SCALE * depth2).float()
    image1 = normalize_image(image1)
    image2 = normalize_image(image2)

    return image1, image2, depth1, depth2

def prepare_images_and_depths_kitti(image1, image2, depth1, depth2, depth_scale=1.0):
    """ padding, normalization, and scaling """
    
    ht, wd = image1.shape[-2:]
    pad_h = (-ht) % 8
    pad_w = (-wd) % 8

    image1 = F.pad(image1, [0,pad_w,0,pad_h], mode='replicate')
    image2 = F.pad(image2, [0,pad_w,0,pad_h], mode='replicate')
    depth1 = F.pad(depth1[:,None], [0,pad_w,0,pad_h], mode='replicate')[:,0]
    depth2 = F.pad(depth2[:,None], [0,pad_w,0,pad_h], mode='replicate')[:,0]

    depth1 = (depth_scale * depth1).float()
    depth2 = (depth_scale * depth2).float()
    image1 = normalize_image(image1.float())
    image2 = normalize_image(image2.float())

    depth1 = depth1.float()
    depth2 = depth2.float()

    return image1, image2, depth1, depth2, (pad_w, pad_h)

def display(img, tau, phi):
    """ display se3 fields """
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.imshow(img[:, :, ::-1] / 255.0)

    tau_img = np.clip(tau, -0.1, 0.1)
    tau_img = (tau_img + 0.1) / 0.2

    phi_img = np.clip(phi, -0.1, 0.1)
    phi_img = (phi_img + 0.1) / 0.2

    ax2.imshow(tau_img)
    ax3.imshow(phi_img)
    plt.show()


@torch.no_grad()
def demo(args):
    import importlib
    RAFT3D = importlib.import_module(args.network).RAFT3D
    model = torch.nn.DataParallel(RAFT3D(args))
    model.load_state_dict(torch.load(args.model), strict=False)

    model.eval()
    model.cuda()

    fx, fy, cx, cy = (1050.0, 1050.0, 480.0, 270.0)
    b = 1.0 # baseline in synthetic blender data
    img1 = cv2.imread('assets/image1.png')
    img2 = cv2.imread('assets/image2.png')
    disp1 = frame_utils.read_gen('assets/disp1.pfm')
    disp2 = frame_utils.read_gen('assets/disp2.pfm')

    # img1 = cv2.imread('/home/christian/Downloads/Sampler/FlyingThings3D/RGB_cleanpass/left/0006.png')
    # img2 = cv2.imread('/home/christian/Downloads/Sampler/FlyingThings3D/RGB_cleanpass/left/0007.png')
    # disp1 = frame_utils.read_gen('/home/christian/Downloads/Sampler/FlyingThings3D/disparity/0006.pfm')
    # disp2 = frame_utils.read_gen('/home/christian/Downloads/Sampler/FlyingThings3D/disparity/0007.pfm')

    # fx, fy, cx, cy = (959.791, 956.9251, 696.0217, 224.1806)
    intrinsics = np.array([959.791, 956.9251, 696.0217, 224.1806])
    # http://www.cvlibs.net/datasets/kitti/setup.php
    b = 0.54 # array([ 0.53267121, -0.00526146,  0.00782809])
    img1 = cv2.imread('datasets/KITTI/testing/image_2/000000_10.png')
    img2 = cv2.imread('datasets/KITTI/testing/image_2/000000_11.png')
    disp1 = cv2.imread('datasets/KITTI/testing/disp_ganet_testing/000000_10.png', cv2.IMREAD_ANYDEPTH) / 256.0
    disp2 = cv2.imread('datasets/KITTI/testing/disp_ganet_testing/000001_10.png', cv2.IMREAD_ANYDEPTH) / 256.0

    d1 = (b * intrinsics[0] / disp1).astype(np.uint16)
    d2 = (b * intrinsics[0] / disp2).astype(np.uint16)
    print("d1", d1.shape, d1.dtype, np.min(d1),np.max(d1))
    cv2.imwrite("d1.png", d1)
    cv2.imwrite("d2.png", d2)

    crop = 80
    img1 = img1[crop:]
    img2 = img2[crop:]
    disp1 = disp1[crop:]
    disp2 = disp2[crop:]
    intrinsics[3] -= crop

    # image1 = np.expand_dims(image1, 0)
    # image2 = np.expand_dims(image2, 0)
    # disp1 = np.expand_dims(disp1, 0)
    # disp2 = np.expand_dims(disp2, 0)
    # intrinsics = np.expand_dims(intrinsics, 0)

    # depth1 = torch.from_numpy(fx / disp1).float().cuda().unsqueeze(0)
    # depth2 = torch.from_numpy(fx / disp2).float().cuda().unsqueeze(0)
    # image1 = torch.from_numpy(img1).permute(2,0,1).float().cuda().unsqueeze(0)
    # image2 = torch.from_numpy(img2).permute(2,0,1).float().cuda().unsqueeze(0)
    # intrinsics = torch.as_tensor([fx, fy, cx, cy]).cuda().unsqueeze(0)

    image1 = torch.from_numpy(img1).float().permute(2,0,1).cuda()
    image2 = torch.from_numpy(img2).float().permute(2,0,1).cuda()
    disp1 = torch.from_numpy(disp1).float().cuda()
    disp2 = torch.from_numpy(disp2).float().cuda()
    intrinsics = torch.from_numpy(intrinsics).float().cuda()

    image1 = image1.unsqueeze(0)
    image2 = image2.unsqueeze(0)
    disp1 = disp1.unsqueeze(0)
    disp2 = disp2.unsqueeze(0)
    intrinsics = intrinsics.unsqueeze(0)

    # img1 = image1[0].permute(1,2,0).cpu().numpy()
    depth1 = DEPTH_SCALE * (b * intrinsics[0,0] / disp1)
    depth2 = DEPTH_SCALE * (b * intrinsics[0,0] / disp2)

    # image1, image2, depth1, depth2 = prepare_images_and_depths(image1, image2, depth1, depth2)
    image1, image2, depth1, depth2, _ = prepare_images_and_depths_kitti(image1, image2, depth1, depth2)
    
    # KITTI in torch.Size([1, 3, 296, 1248]) torch.Size([1, 3, 296, 1248]) torch.Size([1, 296, 1248]) torch.Size([1, 296, 1248]) torch.Size([1, 4])
    print("DEMO in", image1.shape, image2.shape, depth1.shape, depth2.shape, intrinsics.shape)
    Ts = model(image1, image2, depth1, depth2, intrinsics, iters=16)
    
    # compute 2d and 3d from from SE3 field (Ts)
    flow2d, flow3d, _ = pops.induced_flow(Ts, depth1, intrinsics)

    # extract rotational and translational components of Ts
    tau, phi = Ts.log().split([3,3], dim=-1)
    tau = tau[0].cpu().numpy()
    phi = phi[0].cpu().numpy()

    # undo depth scaling
    flow3d = flow3d / DEPTH_SCALE

    display(img1, tau, phi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='raft3d.pth', help='checkpoint to restore')
    parser.add_argument('--network', default='raft3d.raft3d', help='network architecture')
    args = parser.parse_args()

    demo(args)

    
