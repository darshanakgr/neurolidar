import numpy as np
import cv2


def create_depth_mask(depth_image, max_dist=200):
    return np.where(depth_image > max_dist, 0, 1)


def decode_depth_image(depth_image):
    depth_image = depth_image.astype(np.float32)
    depth_image = ((depth_image[:, :, 2] + depth_image[:, :, 1] * 256 + depth_image[:, :, 0] * (256 ** 2)) / (256 ** 3 - 1))
    depth_image *= 1000
    
    return depth_image

def create_event_frame(image_size, events):
    xs = events["x"]
    ys = events["y"]
    pols = events["pol"]
    img = np.zeros(image_size)
    img[ys, xs] = 2 * pols - 1
    return img