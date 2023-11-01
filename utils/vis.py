import numpy as np
import matplotlib.pyplot as plt
import os, copy
from PIL import Image, ImageDraw
import gzip
import cv2

def apply_2dbbox(rgb_obs, bbox_2d):
    # rgb_img = Image.fromarray(rgb_obs, mode="RGBA")
    rgb_img = copy.deepcopy(rgb_obs)
    draw = ImageDraw.Draw(rgb_img)
    draw.rectangle(bbox_2d, fill=None, outline="red")
        # draw.text((bbox_2d[i][0], bbox_2d[i][1]), fill='red')
    # plt.figure(figsize=(12, 12))
    # plt.imshow(rgb_img)
    # plt.show()
    return rgb_img

def apply_3dbbox(rgb_obs, bbox_3d):
    node_idx = np.array(
        [[1, 3], [1, 5], [3, 7], [5, 7], [4, 5], [4, 6], [6, 7], [2, 6], [3, 2], [0, 4], [0, 2], [0, 1]])
    color_idx = np.array(['blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue',
                          'red', 'red', 'red'])
    # rgb_img = Image.fromarray(rgb_obs, mode="RGBA")
    rgb_img = copy.deepcopy(rgb_obs)
    draw = ImageDraw.Draw(rgb_img)
    for j in range(len(node_idx)):
        line = [bbox_3d[node_idx[j, 0]][0], bbox_3d[node_idx[j, 0]][1], bbox_3d[node_idx[j, 1]][0], bbox_3d[node_idx[j, 1]][1]]
        draw.line(line, fill=color_idx[j])
    # plt.figure(figsize=(12, 12))
    # plt.imshow(rgb_img)
    # plt.show()
    return rgb_img

def apply_bbox3d_onebyone(rgb_obs, bbox_3d):
    # edge order and painting color
    node_idx = np.array(
        [[1, 3], [1, 5], [3, 7], [5, 7], [4, 5], [4, 6], [6, 7], [2, 6], [3, 2], [0, 4], [0, 2], [0, 1]])
    color_idx = np.array(['blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue',
                          'red', 'red', 'red'])
    # rgb_img = Image.fromarray(rgb_obs, mode="RGBA")
    rgb_img = copy.deepcopy(rgb_obs)
    draw = ImageDraw.Draw(rgb_img)

    for i in range(len(bbox_3d)):

        bbox = bbox_3d[i]
        # if name[i]=='bed':
        for j in range(len(node_idx)):
            line = [bbox[node_idx[j, 0]][0], bbox[node_idx[j, 0]][1], bbox[node_idx[j, 1]][0], bbox[node_idx[j, 1]][1]]
            draw.line(line, fill=color_idx[j])
        # draw.text((5, 5), text=name[i])

    # plt.figure(figsize=(12, 12))
    # plt.imshow(rgb_img)
    # plt.show()
    return rgb_img

# Apply the given mask to the image.
def apply_mask(rgb_obs, mask, alpha=0.5):
    rgb_img = np.array(copy.deepcopy(rgb_obs))
    for c in range(3):
        color_ratio = np.random.rand()
        rgb_img[:, :, c] = np.where(mask == 1,
                                  rgb_img[:, :, c] * (1 - alpha) + alpha * color_ratio * 255,
                                  rgb_img[:, :, c])

    return Image.fromarray(rgb_img)


def save_img(save_path, rgb_obs):
    # plt.figure(figsize=(12, 12))
    plt.figure()
    plt.imshow(rgb_obs)
    plt.savefig(save_path)


# refer: s://blog.csdn.net/qq_36http265860/article/details/107908179
def plot_depth_map(depth_path, save_path):
    with gzip.GzipFile(depth_path, 'r') as f:
        depth = np.load(f)

    im_color=cv2.applyColorMap(cv2.convertScaleAbs(depth,alpha=15),cv2.COLORMAP_JET)
    img = Image.fromarray(im_color)
    img.save(save_path)


def plot_normal_map(normal_path, save_path):
    with gzip.GzipFile(normal_path, 'r') as f:
        normal = np.load(f)

    im_color = (normal*255).astype(np.uint8)
    img = Image.fromarray(im_color)
    img.save(save_path)
