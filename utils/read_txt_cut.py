
import os

from PIL import Image
import sys

INPUT_DATA = './85test/image/'


INPUT_TXT = './85test/txt/'

OUTPUT_DATA='./85test/output/'



INPUT_DATA_LABEL = './85test/labelxy/'



OUTPUT_DATA_LABEL='./85test/outputlabel/'


INPUT_DATA_LABEL_xx = './85test/labelxyxx/'



OUTPUT_DATA_LABEL_xx='./85test/outputlabelxx/'




def processxy(file):
    with open(INPUT_TXT + file, "r") as f: 
        data = f.read() 
        str_list = data.split()
        tempx1 = float(str_list[0])
        tempy1 = float(str_list[1])

        tempx2 = float(str_list[2])
        tempy2 = float(str_list[3])

        xpoint=(tempx1+tempx2)/2
        ypoint=(tempy1+tempy2)/2


        return (xpoint,ypoint)



def cut_image(x , y, filename ):

    image = Image.open(INPUT_DATA +filename)

    box_list = []

    box = (x-45, y-45, x+45, y+45)
    box_list.append(box)

    image_list = [image.crop(box) for box in box_list]

    return image_list



def cut_label(x , y, filename ):
 
    image = Image.open(INPUT_DATA_LABEL +filename)

    box_list = []

    box = (x-45, y-45, x+45, y+45)
    box_list.append(box)


    image_list = [image.crop(box) for box in box_list]

    return image_list

def cut_label2(x , y, filename ):
    image = Image.open(INPUT_DATA_LABEL_xx +filename)

    box_list = []

    box = (x-45, y-45, x+45, y+45)
    box_list.append(box)

    image_list = [image.crop(box) for box in box_list]

    return image_list


def save_images(image_list,f,path):
    # index = 1
    for image in image_list:
        image.save(path+f)

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


if __name__ == '__main__':


    for root, dirs, files in os.walk(INPUT_TXT):
        for f in files:
            x,y = processxy(f)
            filename = f[0:-4]
            image3=cut_label2(x,y,filename)
            save_images(image3,filename,OUTPUT_DATA_LABEL_xx)
