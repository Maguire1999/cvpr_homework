import time
import cv2
import numpy as np
import os
from image_matching import Image_Matching


class Image_Stitching(Image_Matching):
    def __init__(self):
        super(Image_Stitching, self).__init__()
        self.smoothing_window_size = 200

    def set_smoothing_window(self,img1,img2):
        if self.smoothing_window_size>max(img1.shape[1],img2.shape[1]) \
                or self.smoothing_window_size<min(img1.shape[1]/4,img2.shape[1]/4):
            self.smoothing_window_size = int((img1.shape[1] + img2.shape[1])/4)

    def create_mask(self, img1, img2, version):
        self.set_smoothing_window(img1,img2)

        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if version == 'left_image':
            mask[:, barrier - offset:barrier +
                 offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset:barrier +
                 offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])
    def get_transforming(self, img1 , img2):
        H = self.get_Homography(img1, img2)
        height_img1 = img1.shape[0]
        height_img2 = img2.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2
        transform1 = cv2.warpPerspective(
            img1, H, (width_panorama, height_panorama))
        cv2.imwrite('./results/transform1.jpg', transform1)
        transform2 = cv2.warpPerspective(
            img2, H, (width_panorama, height_panorama))
        cv2.imwrite('./results/transform2.jpg', transform2)
        print("image transforming succeed")

        return transform1,transform2


    def get_stitching(self, img1, img2,with_mask = True):
        panorama1, panorama2 = self.get_transforming(img1,img2)
        panorama1 = np.zeros(
            (img1.shape[0], img1.shape[1] + img2.shape[1], 3))
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1

        if with_mask:
            mask1 = self.create_mask(img1, img2, version='left_image')
        else:
            mask1 = 1
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1
        if with_mask:
            mask2 = self.create_mask(img1, img2, version='right_image')
        else:
            mask2 = 1
        panorama2 = panorama2*mask2

        panorama = panorama1+panorama2
        rows, cols = np.where(panorama[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        stitch = panorama[min_row:max_row, min_col:max_col, :]
        if with_mask:
            cv2.imwrite('./results/stitch.jpg', stitch)
        else:
            cv2.imwrite('./results/stitch_without_mask.jpg', stitch)
        print("image stitching succeed")
        return stitch
        


if __name__ == '__main__':

    path1 = '../data/building1.jpg'
    path2 = '../data/building2.jpg'
    result_dir = './results/'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    Stitcher = Image_Stitching()
    stitch_result = Stitcher.get_stitching(img1, img2)
