import cv2 as cv
import numpy as np


class ImageStitching:
    def __init__(self):
        self.ratio = 0.6
        self.min_match = 10
        self.sift_core = cv.xfeatures2d.SIFT_create()
        self.flann_index_kdtree = 0

    def detection(self, image1, image2):
        # 灰度化
        grey_img1 = cv.cvtColor(image1, cv.COLOR_RGB2GRAY)
        grey_img2 = cv.cvtColor(image2, cv.COLOR_RGB2GRAY)
        cv.imwrite('grey1.jpg', grey_img2)

        # sift提取特征向量
        kp1, des1 = self.sift_core.detectAndCompute(grey_img1, None)
        kp2, des2 = self.sift_core.detectAndCompute(grey_img2, None)
        # points_img1 = cv.drawKeypoints(
        #     image=img1, outImage=img1,
        #     keypoints=kp1,
        #     flags=cv.DRAW_MATCHES_FLAGS_DEFAULT,
        #     color=(51, 163, 236))
        # points_img2 = cv.drawKeypoints(
        #     image=img2, outImage=img2,
        #     keypoints=kp2,
        #     flags=cv.DRAW_MATCHES_FLAGS_DEFAULT,
        #     color=(51, 163, 236))
        # cv.imwrite('points_image1.jpg', points_img1)
        # cv.imwrite('points_image2.jpg', points_img2)
        print('图(a)极值点数： ', len(kp1))
        print('图(b)极值点数： ', len(kp2))
        # k-d树，FLANN寻找匹配点
        indexParams = dict(algorithm=self.flann_index_kdtree, trees=5)
        searchParams = dict(checks=100)
        flann = cv.FlannBasedMatcher(indexParams, searchParams)
        matches = flann.knnMatch(des1, des2, k=2)
        # KNN筛选欧氏距离在0.7内的匹配点
        good_points = []
        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio * n.distance:
                good_points.append((m.trainIdx, m.queryIdx))
                good_matches.append([m])
        matched_image = cv.drawMatchesKnn(image1, kp1, image2, kp2, good_matches, None, flags=2)
        # RANSAC 计算单应矩阵
        h_matrix = None
        if len(good_matches) > self.min_match:
            image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])
            h_matrix, _ = cv.findHomography(image2_kp, image1_kp, cv.RANSAC, 5.0)
        cv.imwrite('matched.jpg', matched_image)
        print('筛选优化后匹配点数： ', len(good_points))
        print(h_matrix)
        return h_matrix

    def get_gradient_mask(self, image1, image2, is_right):
        height_img1 = image1.shape[0]
        width_img1 = image1.shape[1]
        width_img2 = image2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2
        smoothing_window_size = (width_img1 + width_img2) / 6
        offset = int(smoothing_window_size / 2)
        barrier = image1.shape[1] - int(smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if is_right == 0:
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        return cv.merge([mask, mask, mask])

    def get_hat_mask(self, image1, image2, is_right):
        height_img1 = image1.shape[0]
        width_img1 = image1.shape[1]
        width_img2 = image2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2
        mask = np.zeros((height_panorama, width_panorama))
        if is_right == 0:
            mask[:, width_img1:] = np.tile(np.linspace(1, 0.5, width_img2).T, (height_panorama, 1))
            mask[:, :width_img1] = 1
        else:
            mask[:, :width_img1] = np.tile(np.linspace(0.5, 1,width_img1).T, (height_panorama, 1))
            mask[:, :width_img1] = 1
        return cv.merge([mask, mask, mask])

    def blending(self, image1, image2):
        H = self.detection(image1, image2)
        height_img1 = image1.shape[0]
        width_img1 = image1.shape[1]
        width_img2 = image2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self.get_gradient_mask(image1, image2, 0)
        panorama1[0:image1.shape[0], 0:image1.shape[1], :] = image1
        panorama1 *= mask1
        mask2 = self.get_gradient_mask(image1, image2, 1)
        panorama2 = cv.warpPerspective(image2, H, (width_panorama, height_panorama))* mask2
        result = panorama1 + panorama2
        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        cv.imwrite('panorama1.jpg', panorama1)
        cv.imwrite('panorama2.jpg', panorama2)
        cv.imwrite('stitched.jpg', final_result)
        return final_result


if __name__ == '__main__':
    img1 = cv.imread('outL.jpg')
    img2 = cv.imread('outM.jpg')
    stitched = ImageStitching().blending(img1, img2)
    # final = ImageStitching().blending(stitched, img3)
    # ImageStitching().detection(img1, img2)
