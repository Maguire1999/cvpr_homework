import time
import cv2
import numpy as np
import os

class Image_Matching():
    def __init__(self):
        self.ratio = 0.85
        self.min_match = 10
        # self.sift = cv2.SIFT_create()
    def get_sift(self,img,name):
        img_copy = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()  # sift为实例化的sift函数
        kp = sift.detect(img, None)  # 找出图像中的关键点
        img_draw = cv2.drawKeypoints(gray, kp, img_copy)  # 画出图像中的关键点
        kp, dst = sift.detectAndCompute(gray, None)
        path = './results/sift_keypoints_' + str(name) + '.jpg'
        cv2.imwrite(path, img_draw)
        return kp, dst
    def get_Homography(self, img1, img2):
        
        kp1, des1 = self.get_sift(img1,'1')
        kp2, des2 = self.get_sift(img2,'2')
        # kp2, des2 = self.sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches = []
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        img_good_matches = cv2.drawMatchesKnn(
            img1, kp1, img2, kp2, good_matches, None, flags=2)
        img_raw_matches = cv2.drawMatchesKnn(
            img1, kp1, img2, kp2, raw_matches, None, flags=2)
        cv2.imwrite('./results/good_matching.jpg', img_good_matches)
        cv2.imwrite('./results/raw_matching.jpg', img_raw_matches)
        if len(good_points) > self.min_match:
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(
                image2_kp, image1_kp, cv2.RANSAC, 5.0)
            print("image matching succeed")
            return H


if __name__ == '__main__':
    
    path1 = '../data/montain1.jpg'
    path2 = '../data/montain2.jpg'
    result_dir = './results/'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    macher = Image_Matching()
    H = macher.get_Homography(img1, img2)
