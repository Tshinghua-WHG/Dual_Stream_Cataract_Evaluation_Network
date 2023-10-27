from __future__ import division,print_function
import os
import shutil

import numpy as np
from PIL import Image, ImageFilter



path1='/1/'
path2='/2/'



def convert(fname, crop_size):
    img = Image.open(fname)
    debug = 1
    #blurred = img.filter(ImageFilter.BLUR) 
    #ba = np.array(blurred)
    ba = np.array(img)
    h, w, _ = ba.shape
    if debug>0:
        print("h=%d, w=%d"%(h,w))
    if w > 1.2 * h:
        left_max = ba[:, : w // 32, :].max(axis=(0, 1)).astype(int)
        right_max = ba[:, - w // 32:, :].max(axis=(0, 1)).astype(int)
        max_bg = np.maximum(left_max, right_max)
        #这个参数原来是5.
        foreground = (ba > max_bg + 5).astype(np.uint8)
        bbox = Image.fromarray(foreground).getbbox()
 
        if debug>0:
            print(foreground, left_max, right_max, bbox)
        if bbox is None:
            print('bbox none for {} (???)'.format(fname))
        else:
            left, upper, right, lower = bbox
            if right - left < 0.8 * h or lower - upper < 0.8 * h:
                print('bbox too small for {}'.format(fname))
                bbox = None
    else:
        bbox = None
 
    if bbox is None:
        if debug>0:
            print 
        bbox = square_bbox(img)
 
    cropped = img.crop(bbox)
    resized = cropped.resize([crop_size, crop_size])
    return resized
 
def square_bbox(img):
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return (left, upper, right, lower)


def mkdir(path):
    import os

    path = path.strip()
    path = path.rstrip("\\")


    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        print(path + ' Success')
        return True
    else:
        return False


def movefile(src,dst):
    try:
        shutil.copy(src, dst)
    except IOError:
        print("Error: 没有找到文件或读取文件失败")
    else:
        print("内容写入文件成功")

def main():



    for root, dirs, files in os.walk(path1):

        for f in files:
            print(os.path.join(root, f))
            img = convert(path1+f, 2300)
            img.save(path2+f)



if __name__=='__main__':
    main()
