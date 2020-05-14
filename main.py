# Instructions:
# Do not change the output file names, use the helper functions as you see fit

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math



def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Perspective warping")
   print("2 Cylindrical warping")
   print("3 Laplacian perspective warping")
   print("4 Laplacin cylindrical warping")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image1] " + "[path to input image2] " + "[path to input image3] " + "[output directory]")

'''
Detect, extract and match features between img1 and img2.
Using SIFT as the detector/extractor, but this is inconsequential to the user.

Returns: (pts1, pts2), where ptsN are points on image N.
    The lists are "aligned", i.e. point i in pts1 matches with point i in pts2.

Usage example:
    im1 = cv2.imread("image1.jpg", 0)
    im2 = cv2.imread("image2.jpg", 0)
    (pts1, pts2) = feature_matching(im1, im2)

    plt.subplot(121)
    plt.imshow(im1)
    plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
    plt.subplot(122)
    plt.imshow(im2)
    plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
'''
def feature_matching(img1, img2, savefig=False):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches2to1 = flann.knnMatch(des2,des1,k=2)

    matchesMask_ratio = [[0,0] for i in range(len(matches2to1))]
    match_dict = {}
    for i,(m,n) in enumerate(matches2to1):
        if m.distance < 0.7*n.distance:
            matchesMask_ratio[i]=[1,0]
            match_dict[m.trainIdx] = m.queryIdx

    good = []
    recip_matches = flann.knnMatch(des1,des2,k=2)
    matchesMask_ratio_recip = [[0,0] for i in range(len(recip_matches))]

    for i,(m,n) in enumerate(recip_matches):
        if m.distance < 0.7*n.distance: # ratio
            if m.queryIdx in match_dict and match_dict[m.queryIdx] == m.trainIdx: #reciprocal
                good.append(m)
                matchesMask_ratio_recip[i]=[1,0]



    if savefig:
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask_ratio_recip,
                           flags = 0)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,recip_matches,None,**draw_params)

        plt.figure(),plt.xticks([]),plt.yticks([])
        plt.imshow(img3,)
        plt.savefig("feature_matching.png",bbox_inches='tight')

    return ([ kp1[m.queryIdx].pt for m in good ],[ kp2[m.trainIdx].pt for m in good ])

'''
Warp an image from cartesian coordinates (x, y) into cylindrical coordinates (theta, h)
Returns: (image, mask)
Mask is [0,255], and has 255s wherever the cylindrical images has a valid value.
Masks are useful for stitching

Usage example:

    im = cv2.imread("myimage.jpg",0) #grayscale
    h,w = im.shape
    f = 700
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
    imcyl = cylindricalWarpImage(im, K)
'''
def cylindricalWarpImage(img1, K, savefig=False):
    f = K[0,0]

    im_h,im_w = img1.shape

    # go inverse from cylindrical coord to the image
    # (this way there are no gaps)
    cyl = np.zeros_like(img1)
    cyl_mask = np.zeros_like(img1)
    cyl_h,cyl_w = cyl.shape
    x_c = float(cyl_w) / 2.0
    y_c = float(cyl_h) / 2.0
    for x_cyl in np.arange(0,cyl_w):
        for y_cyl in np.arange(0,cyl_h):
            theta = (x_cyl - x_c) / f
            h     = (y_cyl - y_c) / f

            X = np.array([math.sin(theta), h, math.cos(theta)])
            X = np.dot(K,X)
            x_im = X[0] / X[2]
            if x_im < 0 or x_im >= im_w:
                continue

            y_im = X[1] / X[2]
            if y_im < 0 or y_im >= im_h:
                continue

            cyl[int(y_cyl),int(x_cyl)] = img1[int(y_im),int(x_im)]
            cyl_mask[int(y_cyl),int(x_cyl)] = 255


    if savefig:
        plt.imshow(cyl, cmap='gray')
        plt.savefig("cyl.png",bbox_inches='tight')

    return (cyl,cyl_mask)

'''
Calculate the geometric transform (only affine or homography) between two images,
based on feature matching and alignment with a robust estimator (RANSAC).

Returns: (M, pts1, pts2, mask)
Where: M    is the 3x3 transform matrix
       pts1 are the matched feature points in image 1
       pts2 are the matched feature points in image 2
       mask is a binary mask over the lists of points that selects the transformation inliers

Usage example:
    im1 = cv2.imread("image1.jpg", 0)
    im2 = cv2.imread("image2.jpg", 0)
    (M, pts1, pts2, mask) = getTransform(im1, im2)

    # for example: transform im1 to im2's plane
    # first, make some room around im2
    im2 = cv2.copyMakeBorder(im2,200,200,500,500, cv2.BORDER_CONSTANT)
    # then transform im1 with the 3x3 transformation matrix
    out = cv2.warpPerspective(im1, M, (im1.shape[1],im2.shape[0]), dst=im2.copy(), borderMode=cv2.BORDER_TRANSPARENT)

    plt.imshow(out, cmap='gray')
    plt.show()
'''
def getTransform(src, dst, method='affine'):
    pts1,pts2 = feature_matching(src,dst)

    src_pts = np.float32(pts1).reshape(-1,1,2)
    dst_pts = np.float32(pts2).reshape(-1,1,2)

    if method == 'affine':
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)

    if method == 'homography':
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()

    return (M, pts1, pts2, mask)
   
# ===================================================
# ================ Perspective Warping ==============
# ===================================================
def Perspective_warping(img1, img2, img3):
    
    # Transform Image2 and Image3 in plane of Image1.
    
    output_name = sys.argv[5] + "output_homography.png"
    
    # Add padding to allow space around the center image to paste other images.
    img1 = cv2.copyMakeBorder(img1,200,200,500,500, cv2.BORDER_CONSTANT)
    
    # Calculate homography transformation for img2 to img1.
    (M21, pts221, pts121, mask21) = getTransform(img2,img1, 'homography')
    
    # Transformed image of img1 and img2 using homography.
    tranformedImage = cv2.warpPerspective(img2, M21, (img1.shape[1],img1.shape[0]), dst=img1.copy(),borderMode=cv2.BORDER_TRANSPARENT)

    # Calculate homography transformation for img3 to transformed image.
    (M31, pts231, pts131, mask31) = getTransform(img3,tranformedImage, 'homography')
    output_image = cv2.warpPerspective(img3, M31, (tranformedImage.shape[1],tranformedImage.shape[0]),dst=tranformedImage.copy(), borderMode=cv2.BORDER_TRANSPARENT)
    
    # Write output image.
    cv2.imwrite(output_name, output_image)
    return True

def Laplacian_perspective_warping(img1, img2, img3):
    
    output_name = sys.argv[5] + "output_homography_lpb.png"
    
    # Add padding to allow space around the center image to paste other images.
    img1 = cv2.copyMakeBorder(img1,200,200,500,500, cv2.BORDER_CONSTANT)
    
    # Calculate homography transformation for img2 to img1.
    (M21, pts221, pts121, mask21) = getTransform(img2,img1, 'homography')
    
    # Transformed image of img1 and img2 using homography.
    tranformedImage2 = cv2.warpPerspective(img2, M21, (img1.shape[1],img1.shape[0]), dst=img1.copy(),borderMode=cv2.BORDER_TRANSPARENT)

    # Calculate homography transformation for img3 to img1.
    (M31, pts231, pts131, mask31) = getTransform(img3,img1, 'homography')
    tranformedImage3 = cv2.warpPerspective(img3, M31, (img1.shape[1],img1.shape[0]), dst=img1.copy(),borderMode=cv2.BORDER_TRANSPARENT)

    # Combine both transformed images
    # Pass tranformedImage3 as first and tranformedImage2 as second parameter in function because
    # laplacian function blend by taking first half of first image and second hafl of second image.
    output_image = Laplacian_blending(tranformedImage3, tranformedImage2)
    
    cv2.imwrite(output_name, output_image)

    return True

def Laplacian_blending(img1, img2):
    
    levels = 3
    # generating Gaussian pyramids for both images
    gpImg1 = [img1.astype('float32')]
    gpImg2 = [img2.astype('float32')]
    for i in range(levels):
        img1 = cv2.pyrDown(img1)   # Downsampling using Gaussian filter
        gpImg1.append(img1.astype('float32'))
        img2 = cv2.pyrDown(img2)
        gpImg2.append(img2.astype('float32'))

    # Generating Laplacin pyramids for both images
    lpImg1 = [gpImg1[levels]]
    lpImg2 = [gpImg2[levels]]

    for i in range(levels,0,-1):
        # Upsampling and subtracting from upper level Gaussian pyramid image to get Laplacin pyramid image
        tmp = cv2.pyrUp(gpImg1[i]).astype('float32')
        tmp = cv2.resize(tmp, (gpImg1[i-1].shape[1],gpImg1[i-1].shape[0]))
        lpImg1.append(np.subtract(gpImg1[i-1],tmp))

        tmp = cv2.pyrUp(gpImg2[i]).astype('float32')
        tmp = cv2.resize(tmp, (gpImg2[i-1].shape[1],gpImg2[i-1].shape[0]))
        lpImg2.append(np.subtract(gpImg2[i-1],tmp))

    laplacianList = []
    for lImg1,lImg2 in zip(lpImg1,lpImg2):
        rows,cols = lImg1.shape
        # Merging first and second half of first and second images respectively at each level in pyramid
        mask1 = np.zeros(lImg1.shape)
        mask2 = np.zeros(lImg2.shape)
        mask1[:, 0:int(cols/ 2)] = 1
        mask2[:, int(cols / 2):] = 1

        tmp1 = np.multiply(lImg1, mask1.astype('float32'))
        tmp2 = np.multiply(lImg2, mask2.astype('float32'))
        tmp = np.add(tmp1, tmp2)
        
        laplacianList.append(tmp)
    
    img_out = laplacianList[0]
    for i in range(1,levels+1):
        img_out = cv2.pyrUp(img_out)   # Upsampling the image and merging with higher resolution level image
        img_out = cv2.resize(img_out, (laplacianList[i].shape[1],laplacianList[i].shape[0]))
        img_out = np.add(img_out, laplacianList[i])
    
    np.clip(img_out, 0, 255, out=img_out)
    return img_out.astype('uint8')

# ===================================================
# =============== Cynlindrical Warping ==============
# ===================================================

# Stich two images using a mask. Copy warpedImage in baseImage only if value in mask is 255.
def Stitch_images(baseImage, warpedImage, warpedImageMask):
    rows,cols = baseImage.shape
    for r in range(0,rows):
        for c in range(0,cols):
            if warpedImageMask[r][c] == 255:
                baseImage[r][c] = warpedImage[r][c]
    return baseImage

def Cylindrical_warping(img1, img2, img3):
    
    # Transform Image2 and Image3 in plane of Image1.
    
    # Find cylindrical co-ordinates ofr images.
    h,w = img1.shape
    f = 425
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
    (img1cyl, mask1) = cylindricalWarpImage(img1, K)  #Returns: (image, mask)
    (img2cyl, mask2) = cylindricalWarpImage(img2, K)
    (img3cyl, mask3) = cylindricalWarpImage(img3, K)
    
    # Add padding to allow space around the center image to paste other images.
    img1cyl = cv2.copyMakeBorder(img1cyl,50,50,300,300, cv2.BORDER_CONSTANT)
    
    # Calculate Affine transformation to transform image2 in plane of image1.
    (M21, pts2, pts1, mask4) = getTransform(img2cyl, img1cyl, 'affine')
    
    # Transform image2 and mask2 in plane of image1.
    transformedImage2 = cv2.warpAffine(img2cyl, M21, (img1cyl.shape[1],img1cyl.shape[0]))
    
    transformedMask2 = cv2.warpAffine(mask2, M21, (img1cyl.shape[1],img1cyl.shape[0]))
    
    # Stich image1 and image2 using mask.
    transformedImage21 = Stitch_images(img1cyl, transformedImage2, transformedMask2)
    
    # Transformation image3 in plane of transformed image.
    (M31, pts2, pts1, mask5) = getTransform(img3cyl, transformedImage21, 'affine')
    transformedImage3 = cv2.warpAffine(img3cyl, M31, (transformedImage21.shape[1],transformedImage21.shape[0]))
    transformedMask3 = cv2.warpAffine(mask3, M31, (transformedImage21.shape[1],transformedImage21.shape[0]))
    
    # Stich image3 and transformed image using mask.
    output_image = Stitch_images(transformedImage21, transformedImage3, transformedMask3)
    
    # Write out the result
    output_name = sys.argv[5] + "output_cylindrical.png"
    cv2.imwrite(output_name, output_image)
    
    return True

def Laplacian_cylindrical_warping(img1, img2, img3):
    
    # Write your codes here
    h,w = img1.shape
    f = 425
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
    (img1cyl, mask1) = cylindricalWarpImage(img1, K)  #Returns: (image, mask)
    (img2cyl, mask2) = cylindricalWarpImage(img2, K)
    (img3cyl, mask3) = cylindricalWarpImage(img3, K)
    
    # Add padding to allow space around the center image to paste other images.
    img1cyl = cv2.copyMakeBorder(img1cyl,50,50,300,300, cv2.BORDER_CONSTANT)
    
    # Calculate Affine transformation to transform image2 in plane of image1.
    (M21, pts2, pts1, mask4) = getTransform(img2cyl, img1cyl, 'affine')

    # Transform image2 and mask2 in plane of image1.
    transformedImage2 = cv2.warpAffine(img2cyl, M21, (img1cyl.shape[1],img1cyl.shape[0]))
    
    transformedMask2 = cv2.warpAffine(mask2, M21, (img1cyl.shape[1],img1cyl.shape[0]))
    
    # Stich image1 and image2 using mask.
    transformedImage21 = Stitch_images(img1cyl, transformedImage2, transformedMask2)
    
    # Transformation image3 in plane of image1.
    (M31, pts2, pts1, mask5) = getTransform(img3cyl, img1cyl, 'affine')

    transformedImage3 = cv2.warpAffine(img3cyl, M31, (img1cyl.shape[1],img1cyl.shape[0]))
    transformedMask3 = cv2.warpAffine(mask3, M31, (img1cyl.shape[1],img1cyl.shape[0]))
    transformedImage31 = Stitch_images(transformedImage21, transformedImage3, transformedMask3)
    
    # Combine both transformed images using Laplacian.
    output_image = Laplacian_blending(transformedImage31, transformedImage21)
    
    output_name = "output_cylindrical_lpb.png"
    cv2.imwrite(output_name, output_image)
    
    return True


if __name__ == '__main__':
    input_image1 = cv2.imread('input1.png', 0)
    input_image2 = cv2.imread('input2.png', 0)
    input_image3 = cv2.imread('input3.png', 0)
    Laplacian_cylindrical_warping(input_image1, input_image2,input_image3)
   # question_number = -1
   #
   # # Validate the input arguments
   # if (len(sys.argv) != 6):
   #    help_message()
   #    sys.exit()
   # else:
   #     question_number = int(sys.argv[1])
   #     if (question_number > 4 or question_number < 1):
   #        print("Input parameters out of bound ...")
   #        sys.exit()
   #

   #
   # function_launch = {
   # 1 : Perspective_warping,
   # 2 : Cylindrical_warping,
   # 3 : Laplacian_perspective_warping,
   # 4 : Laplacian_cylindrical_warping,
   # }
   #
   # # Call the function
   # function_launch[question_number](input_image1, input_image2, input_image3)
