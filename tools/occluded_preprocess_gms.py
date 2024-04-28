import cv2
import numpy as np

def stitch_images(img1, img2):
    # 初始化SIFT检测器
    sift = cv2.xfeatures2d.SIFT_create()

    # 找到关键点和描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 创建BFMatcher对象
    bf = cv2.BFMatcher()

    # 匹配描述符
    matches = bf.knnMatch(des1, des2, k=2)

    # 应用比率测试
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    # 计算基础矩阵
    pts1 = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    pts2 = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # 我们只选择内点
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    # 计算极线
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1,3)

    # 使用GMS进行匹配
    matcher = cv2.xfeatures2d.matchGMS(img1.shape[:2], img2.shape[:2], kp1, kp2, good, withRotation=True, withScale=True, thresholdFactor=6.0)

    # 计算视角变换矩阵
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matcher ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matcher ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    # 使用视角变换矩阵融合图像
    result = cv2.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2

    return result

# 读取图像
img1 = cv2.imread('/data3/dn/project/mmocr/data/Occluded_RoadText/val/val_000.png')
img2 = cv2.imread('/data3/dn/project/mmocr/data/Occluded_RoadText/val_supplementary/val_sup_000.png')
img3 = cv2.imread('/data3/dn/project/mmocr/data/Occluded_RoadText/val_supplementary/val_sup_001.png')

# 融合图像
result = stitch_images(img1, img2)
result = stitch_images(result, img3)

# 保存结果图像
cv2.imwrite('result_gms.jpg', result)