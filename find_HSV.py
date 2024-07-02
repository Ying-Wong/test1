import cv2
import numpy as np

# 加载图像
image_path = "color_image.png"
image = cv2.imread(image_path)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 创建窗口
cv2.namedWindow('Segmented Image')


#
# 定义绿色范围的阈值
# lower_green = np.array([0, 0, 0])
# upper_green = np.array([129, 62, 255])
# lower_green = np.array([127, 63, 0])
# upper_green = np.array([179, 255, 255])
# 定义回调函数
def nothing(x):
    pass


# 创建trackbars来调整HSV的上下限
cv2.createTrackbar('H Lower', 'Segmented Image', 0, 179, nothing)
cv2.createTrackbar('S Lower', 'Segmented Image', 0, 255, nothing)
cv2.createTrackbar('V Lower', 'Segmented Image', 0, 255, nothing)
cv2.createTrackbar('H Upper', 'Segmented Image', 179, 179, nothing)
cv2.createTrackbar('S Upper', 'Segmented Image', 255, 255, nothing)
cv2.createTrackbar('V Upper', 'Segmented Image', 255, 255, nothing)

while True:
    # 获取当前的trackbar位置
    h_lower = cv2.getTrackbarPos('H Lower', 'Segmented Image')
    s_lower = cv2.getTrackbarPos('S Lower', 'Segmented Image')
    v_lower = cv2.getTrackbarPos('V Lower', 'Segmented Image')
    h_upper = cv2.getTrackbarPos('H Upper', 'Segmented Image')
    s_upper = cv2.getTrackbarPos('S Upper', 'Segmented Image')
    v_upper = cv2.getTrackbarPos('V Upper', 'Segmented Image')

    # 定义HSV的上下限
    lower_bound = np.array([h_lower, s_lower, v_lower])
    upper_bound = np.array([h_upper, s_upper, v_upper])

    # 创建掩码
    mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)

    # 显示结果
    cv2.imshow('Segmented Image', result)

    # 按下'Esc'键退出
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 销毁所有窗口
cv2.destroyAllWindows()
