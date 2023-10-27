import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


INPUT_DATA = './82pre/clahe/'
OUTPUT1 = './82pre/gaussian/'




def gaussian(image,path):

    mri_img = cv2.imread(path+image)

    blur = cv2.GaussianBlur(mri_img, (3, 3), 1.3)

    cv2.imwrite(OUTPUT1+image, blur)


def main():

    for root, dirs, files in os.walk(INPUT_DATA):

        for f in files:
            gaussian(f,INPUT_DATA)






if __name__=='__main__':
    main()
