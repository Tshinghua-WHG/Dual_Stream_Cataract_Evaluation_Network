import math

from cv2 import cv2
import numpy as np
import  os



#entropy
def entropy(img):

    out = 0
    # print(img.shape)
    # print(np.shape(img)[0])
    # print(np.shape(img)[1])

    count = np.shape(img)[0]*np.shape(img)[1]
    print(count)

    print(np.array(img).flatten())
    p = np.bincount(np.array(img).flatten())
    print(p)
    for i in range(0, len(p)):
        if p[i]!=0:
            out-=p[i]*math.log(p[i]/count)/count
            print(out)
    return out




def get_big_gray(img):

    graybig=0

    height = img.shape[0]
    weight = img.shape[1]
    # # channels = gray.shape[2]
    # print("weight : %s, height : %s, channel :" % (weight, height))
    # gaodedushu = []
    # count = 0
    for row in range(height): 
        # temp = 0
        for col in range(weight): 

            # pv = graysp[row, col]
            pv2 = img[row, col]
            # print(pv2)
            if pv2 > graybig:
                #     temp=128

                graybig = pv2
    return graybig







def contrast(img0):
    # img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    m, n = img0.shape
    img1_ext = cv2.copyMakeBorder(img1, 1, 1, 1, 1, cv2.BORDER_REPLICATE) / 1.0 
    rows_ext, cols_ext = img1_ext.shape
    b = 0.0
    for i in range(1, rows_ext - 1):
        for j in range(1, cols_ext - 1):
            b += ((img1_ext[i, j] - img1_ext[i, j + 1]) ** 2 + (img1_ext[i, j] - img1_ext[i, j - 1]) ** 2 +
                  (img1_ext[i, j] - img1_ext[i + 1, j]) ** 2 + (img1_ext[i, j] - img1_ext[i - 1, j]) ** 2)

    cg = b / (4 * (m - 2) * (n - 2) + 3 * (2 * (m - 2) + 2 * (n - 2)) + 2 * 4)
    # print(cg)
    return cg





if __name__ == '__main__':


    path_c = "./1020/entro/"


    for root, dirs, filename in os.walk(path_c):
        # print(type(grad_cam))
        # 	print(filename)
        # 	filenamelist.append(filename)
        #
        for s in filename:
            imageid = s.split(".")[0]
            print(imageid)
            img1 = cv2.imread(path_c  + s)
            # height, width = img1.shape[:2]
            # cv2.imshow('image1', img1
            # shape = np.shape(img1)
            # print(shape)

            # img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

            con1=entropy(img1)
            print(imageid+" " + str(con1))
