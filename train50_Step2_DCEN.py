import os
import sys
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from utils_loss import FocalLoss
from fusion_data import Build_MultiModel_ShareBackbone
from dataloader.image_transforms import Image_Transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import DataParallel
import torch.nn.init as nn_init
from os import path
from PIL import Image
import numpy as np
import pandas as pd
import datetime
import time




def writefile(name, list):
    # print(list)
    f = open(name+'.txt', mode='w')  
    # f.write(Loss_list)  
    for i in range(len(list)):
        s = str(list[i]).replace('{', '').replace('}', '').replace("'", '').replace(':', ',') + '\n'
        f.write(s)
    f.close()



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






path_join = path.join



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, im_dir, im_names, im_labels, im_extra,im_path,im_transforms=None):
        self.im_dir = im_dir
        self.im_labels = im_labels
        self.im_names = im_names
        self.im_path_head=im_path

        self.im_extra=im_extra
        if im_transforms:
            self.im_transforms = im_transforms
        else:
            self.im_transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):

        return len(self.im_labels)

    def __getitem__(self, idx):

        im_file = os.path.join(self.im_dir,str(self.im_path_head[idx]),self.im_names[idx])
        im = Image.open(im_file).convert('RGB')
        im = self.im_transforms(im)
        return im, self.im_labels[idx], self.im_path_head[idx],self.im_extra[idx]


 

def load_data(label_path, train_lists, img_path,classes,
              batchsize, im_transforms,type):   


    train_sets = []
    train_loaders = []




    for train_list in train_lists:
        full_path_list = path_join(label_path,train_list)
        df = pd.read_csv(full_path_list)
        im_names = df['IMAGE'].to_numpy()
        im_labels=df['TYPE'].to_numpy()
        im_path=df['ID'].to_numpy()
        df[classes].iloc[:, 0] /= 100

        im_extra = torch.tensor(df[classes].to_numpy(), dtype=torch.float)

    
        train_sets.append(CustomDataset(img_path, im_names, im_labels ,im_extra, im_path,im_transforms))
        train_loaders.append(torch.utils.data.DataLoader(train_sets[-1], batch_size=batchsize, shuffle=True,num_workers=8))
        print('Size for {0} = {1}'.format(train_list, len(im_names)))


    return train_loaders[0]



def main():

    taskname='_cataract_withtype'

    #   
    # We have tried different schemes to fuse cataract type features and grading features here, 
    # but due to the limitations of data volume, the performance differences of different schemes are not significant
    # resnet50_mean_aggregation resnet50_dim_reduction  resnet50_threshold_reduction  
    backbone="resnet50_mean_aggregation"
    

    #CrossEntropyLoss  FocalLoss
    loss_type="FocalLoss"
    
    num_class=3

    path='/root/work2023/DCEN/'+taskname




    mkdir(path)
    save_path_best = path+'/resNet50'+taskname+'_best.pth'
    save_path_final = path+'/resNet50'+taskname+'_final.pth'

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
 


 

 
 
    IMAGE_PATH = '/vepfs/gwh_cataract/'


    TRAIN_LISTS = ['solo_train.csv']
    TEST_LISTS = ['solo_test.csv']
    CLASSES = ['grade']
    label_path='/root/work2023/DCEN/data_set/label'

    train_transforms = Image_Transforms(mode='train', dataloader_id=1, square_size=256, crop_size=224).get_transforms()
    val_transforms = Image_Transforms(mode='val', dataloader_id=1, square_size=256, crop_size=224).get_transforms()



    batch_size = 32

    validate_loader = load_data(label_path, TEST_LISTS, IMAGE_PATH,CLASSES, batch_size, val_transforms,'test')

    train_loader = load_data(label_path, TRAIN_LISTS, IMAGE_PATH,CLASSES, batch_size, train_transforms,'train')



    dfres = pd.read_csv(path_join(label_path, TEST_LISTS[0]))
    val_num=dfres.shape[0]

    dftrain = pd.read_csv(path_join(label_path, TRAIN_LISTS[0]))

    train_num=dftrain.shape[0]



    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))


    net = Build_MultiModel_ShareBackbone(backbone=backbone,num_classes=num_class)

 
    #lock the cataract types features
    for param in net.types.parameters():
        param.requires_grad = False




    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net, device_ids=[0,1,2,3,4,5,6,7])



    net.to(device)


    print(net)


    if loss_type=="FocalLoss":
        loss_function = FocalLoss(class_num=2, gamma=2)
    else:
        loss_function = nn.CrossEntropyLoss()
        # loss_function = nn.BCELoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)





    Loss_list = []
    Loss_list_val = []

    Accuracy_list = []
    Accuracy_list_val = []











    epochs =100
    best_acc = 0.0
    train_steps = len(train_loader)

    val_steps=len(validate_loader)
    start_time=time.time()

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_acc = 0.0  #
        val_loss=0.0




        train_bar = tqdm(train_loader)


        for count, (data, non_im_input, target) in enumerate(train_bar):

            optimizer.zero_grad()


            non_im_input = non_im_input.to(device)
            target = target.to(device)
            target=target.view(-1)

            logits = net(data, non_im_input)


            loss = loss_function(logits, target.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
            predict_y = torch.max(logits, dim=1)[1]
            train_acc += torch.eq(predict_y, target.to(device)).sum().item()
        train_accurate = train_acc/train_num
        Accuracy_list.append(train_accurate)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        predict_all=[]
        gt_all=[]

        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_nonimage,val_labels = val_data

                val_labels=val_labels.view(-1)

                outputs = net(val_images.to(device),val_nonimage.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                # print(predict_y)



                loss_val = loss_function(outputs, val_labels.to(device))
                val_loss+=loss_val.item()

                gt_all.extend(val_labels.tolist())
                predict_all.extend(predict_y.tolist())


                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        report = classification_report(gt_all, predict_all)
        print(report)


        Loss_list_val.append(val_loss / val_steps)
        Accuracy_list_val.append( val_accurate)
        Loss_list.append(running_loss / train_steps)





        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path_best)
        if epoch==epochs-1:
            torch.save(net.state_dict(), save_path_final)

    print('Finished Training')





    x1 = range(0, epochs)
    y1 = Accuracy_list
    plt.subplot(1, 2, 1)
    plt.plot(x1, y1, '-',label="Train Accuracy")
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