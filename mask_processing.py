import cv2
import numpy as np
import os
import glob
number =0
def composeImage(img2):
    global number
    src2 = cv2.imread(img2, cv2.IMREAD_UNCHANGED)
    file_name = os.path.basename(img2)
    path, dirname = os.path.split(os.path.dirname(img2))
    print(os.path.join(path, "elastic_all2/","%01d"%number + ".png"))
    src1 = cv2.imread(os.path.join(path, "elastic_all2/", "%01d"%number + ".png"), cv2.IMREAD_COLOR)
    #src1 = cv2.imread("C:/Users/jhkimMultiGpus2080/Desktop/DeepCrack_Test/DeepCrack2/elastic(rotate_45)/*.png")

    #file_name, ext = os.path.splitext(os.path.basename(img2))  # basename : 기본 이름(base name)을 반환, splitest: 확장자 부분과 그 외의 부분으로 나누기
    #dir_name = os.path.dirname(img1)  # 디렉토리명
    #path, dir_name = os.path.split(dir_name)  # 디렉터리 부분과 파일 부분으로 나누기

    src1 = cv2.resize(src1, (227, 227))
    src2 = cv2.resize(src2, (227, 227))
    src1 = np.array(src1, dtype=np.uint8)
    src2 = np.array(src2, dtype=np.uint8)  # mask
    rows, cols, ch = src2.shape
    #roi = src1[0:rows, 0:cols]
    img2gray = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 230, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    src1_bg = cv2.bitwise_and(src1, src1, mask=mask)
    src2_fg = cv2.bitwise_and(src2, src2, mask=mask)
    bgrLower = np.array([255, 255, 255])
    bgrUpper = np.array([255, 255, 255])
    result = cv2.add(src1_bg, src2_fg)
    # cv2.imshow("t", result)
    # cv2.waitKey(0)

    img_mask1 = cv2.inRange(result, bgrLower, bgrUpper)
    img_mask2 = cv2.inRange(result, bgrLower, bgrUpper)
    bgrnew = result.copy()

    bgrnew[img_mask1 > 255] = (52, 52, 52)

    bgrnew2 = bgrnew.copy()
    bgrnew2[img_mask2 > 0] = (52, 52, 52)
    # fgbg = cv2.createBackgroundSubtractorMOG2()
    # bgrnew = fgbg.apply(bgrnew)
    cv2.imwrite("C:/Users/jhkimMultiGpus2080/Desktop/DeepCrack_Test/DeepCrack3/compose_12_0/" +"%01d"%number + ".jpg", bgrnew2)
    number += 1


if __name__ == "__main__":
    #pwd = "C:/Users/jhkimMultiGpus2080/Desktop/DeepCrack_Test/DeepCrack3/elastic(rotate_0)/*.png"
    pwd1 = "C:/Users/jhkimMultiGpus2080/Desktop/DeepCrack_Test/DeepCrack3/negative/*.jpg"

    #image_list = glob.glob(pwd)
    image_list1 = glob.glob(pwd1)
    print(len(image_list1))


    for img1 in image_list1:

        composeImage(img1)
    #cv2.imshow('k', ang)
    #cv2.waitKey(0)

    # k=0
    # for k in range(4):
    #     for img1 in image_list:
    #         fname = "{}.png".format("{0:04d}".format(k))
    #         for img2 in image_list1:
    #
    #             ang = composeImage(img1, img2)
    #             k += 1
    #cv2.imwrite("C:/Users/jhkimMultiGpus2080/PycharmProjects/Crack_Net" + "/COMPOSE4_227/" + fname,ang)

