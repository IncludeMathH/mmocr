import cv2
import numpy as np

# 读取图像
img1 = cv2.imread('/data3/dn/project/mmocr/data/Occluded_RoadText/val/val_000.png')
img2 = cv2.imread('/data3/dn/project/mmocr/data/Occluded_RoadText/val_supplementary/val_sup_000.png')
img3 = cv2.imread('/data3/dn/project/mmocr/data/Occluded_RoadText/val_supplementary/val_sup_001.png')

# 初始化ORB检测器
orb = cv2.ORB_create()

# 创建BFMatcher对象
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 初始化结果图像
result = np.zeros_like(img1)

# 对每个颜色通道进行处理
for i in range(3):
    # 找到关键点和描述符
    kp1, des1 = orb.detectAndCompute(img1[:,:,i], None)
    kp2, des2 = orb.detectAndCompute(img2[:,:,i], None)

    # 匹配描述符
    matches = bf.match(des1, des2)

    # 根据距离排序
    matches = sorted(matches, key = lambda x:x.distance)

    # 计算视角变换矩阵
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    # 使用视角变换矩阵融合图像
    result[:,:,i] = cv2.warpPerspective(img1[:,:,i], M, (img1.shape[1], img1.shape[0]))

# 重复上述步骤，将融合后的图像与第三张图像进行匹配和融合
for i in range(3):
    kp1, des1 = orb.detectAndCompute(result[:,:,i], None)
    kp3, des3 = orb.detectAndCompute(img3[:,:,i], None)

    matches = bf.match(des1, des3)
    matches = sorted(matches, key = lambda x:x.distance)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp3[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    result[:,:,i] = cv2.warpPerspective(result[:,:,i], M, (img1.shape[1], img1.shape[0]))

# 在融合后的图像上进行目标检测
# 这里需要一个预训练的目标检测模型，例如YOLO，SSD，Faster R-CNN等
# 这部分代码需要根据所使用的模型和库进行编写
cv2.imwrite('result.jpg', result)
