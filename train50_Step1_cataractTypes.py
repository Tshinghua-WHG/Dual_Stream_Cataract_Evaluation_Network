import os
import sys
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from resnet import resnet50


import numpy as np


import matplotlib.pyplot as plt

from torch.nn import DataParallel

import torch.nn.init as nn_init



def writefile(name, list):
    # print(list)

    f = open(name+'.txt', mode='w')  # 打开文件，若文件不存在系统自动创建。
    # f.write(Loss_list)  # write 写入
    for i in range(len(list)):
        s = str(list[i]).replace('{', '').replace('}', '').replace("'", '').replace(':', ',') + '\n'
        f.write(s)
    f.close()



def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        # print(path + ' 目录已存在')
        return False




def main():

    taskname='_cataract_type4'




    path='./'+taskname

    mkdir(path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("using {} device.".format(device))
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path


    image_path='/root/work2023/deep-learning-for-image-processing-master/data_set/cataract-pro-fixed'


    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))


    net = resnet50()


    net_dict = net.state_dict()
    predict_model = torch.load("./resnet50-pre.pth")
    state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}# 寻找网络中公共层，并保留预训练参数
    net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
    net.load_state_dict(net_dict)


    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 3)



    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net, device_ids=[0,1,2,3])



    net.to(device)


    print(net)



    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)





    #-----------------添加下
    # 定义4个数组
    Loss_list = []
    Loss_list_val = []

    Accuracy_list = []
    Accuracy_list_val = []

    #-----------------添加上













    epochs = 100
    best_acc = 0.0
    save_path = path+'/resNet50'+taskname+'.pth'
    train_steps = len(train_loader)



    #-----------------添加下

    val_steps=len(validate_loader)
    #-----------------添加上



    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0

        # -----------------添加下

        train_acc = 0.0  #

        val_loss=0.0

        # -----------------添加上




        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            # newimg=np.concatenate((images, images), axis=2) # axes are 0-indexed, i.e. 0, 1, 2


            logits = net(images.to(device))  #这个就是结果
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

            # -----------------添加下

            # print(logits)
            predict_y = torch.max(logits, dim=1)[1]

            train_acc += torch.eq(predict_y, labels.to(device)).sum().item()


            # -----------------添加上

        # -----------------添加下
        train_accurate = train_acc/train_num
        Accuracy_list.append(train_accurate)


        # -----------------添加上




        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch

        predict_y_all=[]

        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                # print(predict_y)

                # -----------------添加下


                loss_val = loss_function(outputs, val_labels.to(device))
                val_loss+=loss_val.item()

                # -----------------添加上


                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
        # print("yanzheng")
        # print(acc)
        # print(val_num)
        val_accurate = acc / val_num




        # -----------------添加下

        Loss_list_val.append(val_loss / val_steps)

        Accuracy_list_val.append( val_accurate)
        Loss_list.append(running_loss / train_steps)

        # -----------------添加上





        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))







        # if val_accurate > best_acc:
        #     best_acc = val_accurate
        #     torch.save(net.state_dict(), save_path)
        torch.save(net.state_dict(), save_path)

    print('Finished Training')





    x1 = range(0, epochs)
    y1 = Accuracy_list
    plt.subplot(1, 2, 1)
    plt.plot(x1, y1, '-',label="Train Accuracy")


    # y2=Accuracy_list_val
    # plt.plot(x1, y2, '-',label="Test Accuracy")



    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()




    plt.subplot(1, 2, 2)
    x2 = range(0, epochs)
    y3 = Loss_list

    # y4=Loss_list_val

    plt.plot(x2, y3, '-', label="Train Loss")
    # plt.plot(x2, y4, '-', label="Test Loss")


    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.legend()

    plt.show()
    plt.savefig(path+'/'+"loss"+taskname+".jpg")



    writefile(path+'/'+"Loss_list"+taskname, Loss_list)

    writefile(path+'/'+"Loss_list_val"+taskname, Loss_list_val)
    writefile(path+'/'+"Accuracy_list"+taskname, Accuracy_list)
    writefile(path+'/'+"Accuracy_list_val"+taskname, Accuracy_list_val)








if __name__ == '__main__':
    main()