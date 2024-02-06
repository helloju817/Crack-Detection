
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2
number = 0

#미시적,거시적 관점 통합
def elastic_rotate__affine(img, alpha, sigma,rotate, alpha_affine, random_state=None):
    img1 = cv2.imread(img)
    global number
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = img1.shape
    shape_size = shape[:2]
    #print(shape_size)  # (384 , 544) : 이미지 사이즈
    # Random affine
    center_square = np.float32(shape_size) // 2

    # center_square : 이미지 중앙 값 : [192. 272.]
    # min(shape_size) : 384 : 이미지 사이즈 가로 세로 중 작은 값=> 여기서는 세로
    # square_size : 128=> 세로 사이즈 3등분
    square_size = min(shape_size) // 3

    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)

    # Cv2.GetAffineTransform(변환 전 픽셀 좌표(pts1), 변환 후 픽셀 좌표(pts2))
    result = cv2.getAffineTransform(pts1, pts2)
    # Cv2.WarpAffine(원본 배열, 결과 배열, 행렬, 결과 배열의 크기, 보간법, 테두리 외삽법, 테두리 색상)
    img2 = cv2.warpAffine(img1, result, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    height, width, channel = img1.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotate, 1)
    img3 = cv2.warpAffine(img2, matrix, (width, height))
    # Cv2.WarpAffine(원본 배열, 결과 배열, 행렬, 결과 배열의 크기, 보간법, 테두리 외삽법, 테두리 색상)
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return map_coordinates(img3, indices, order=1, mode='reflect').reshape(shape)


if __name__ == "__main__":
    import cv2
    import os, glob
    import random
    pwd = "C:/Users/jhkimMultiGpus2080/Desktop/DeepCrack_Test/DeepCrack3/elastic_all/*.png"
    image_list1 = glob.glob(pwd)

    for img1 in image_list1:
        img4 = elastic_rotate__affine(img1, 45, 5, 50, 40)  # 10, 150, 0, 50
        cv2.imwrite("C:/Users/jhkimMultiGpus2080/Desktop/DeepCrack_Test/DeepCrack3/elastic_all6/" + str(
            number) + ".png", img4)
        number += 1