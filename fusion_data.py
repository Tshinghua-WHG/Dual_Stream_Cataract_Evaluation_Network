import torch
import torch.nn as nn
# from fusion.segating import SEGating
# from fusion.average import project,SegmentConsensus
# # from fusion.segating import SEGating
# # from fusion.segating import SEGating
# from fusion.nextvlad import NextVLAD


# from models.cv_models.swin_transformer import swin
# from models.cv_models.swin_transformer_v2 import swinv2
# # from models.cv_models.resnest import resnest50, resnest101
# from models.cv_models.convnext import convnext_tiny, convnext_small, convnext_base
from resnet import resnet50,resnet50_single_cbma

# from models.FIT_Net import FITNet

# from model3d import resnet



class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
        
# img+img share backbone
class Build_MultiModel_ShareBackbone(nn.Module):
    def __init__(self, backbone='resnest50', input_dim=2048, num_classes=2, pretrained_modelpath='None'):
        super().__init__()

        self.num_classes = num_classes
        self.backbone=backbone

        self.severity = resnet50()
        self.severity.fc = Identity()
        self.types = resnet50(num_classes=3)
        model_types_path = "/DCEN/checkpoints/_cataract_type4/resNet50_cataract_type4.pth"
        self.types.load_state_dict(torch.load(model_types_path),strict=False)


        if backbone!='resnet50_threshold_reduction':
            self.types.fc = Identity()
    
        self.lg_mean = torch.nn.Linear(in_features=2048, out_features=self.num_classes)
        self.lg_reduction = torch.nn.Linear(in_features=2048*2, out_features=self.num_classes)
        self.lg_threshold = torch.nn.Linear(in_features=(2048+3), out_features=self.num_classes)

        print('init model:', backbone)




    def forward(self, image):


        severity_features = self.severity(image)
        types_features = self.types(image)
        all=[]
        all.append(severity_features)
        if self.backbone=='resnet50_mean_aggregation':

            all.append(types_features)


            all_avg_feats = torch.stack(all, dim=1)
            all_summed_feats = torch.sum(all_avg_feats, dim=1) 
            all_avg_feats_mean = all_summed_feats / len(all) 

            res = self.lg_mean(all_avg_feats_mean)

        elif self.backbone=='resnet50_dim_reduction':
            all.append(types_features)

            all_output = torch.cat(all, -1) # b, c1+c2
            res = self.lg_reduction(all_output)

        else:
            threshold=0.9
            types_features[types_features >= threshold] = 1
            types_features[types_features < threshold] = 0
            all.append(types_features)
            all_output = torch.cat(all, -1) # b, c1+c2
            res = self.lg_threshold(all_output)


        return res




# from train_multi_avg import load_data_multi
# from dataloader.image_transforms import Image_Transforms
# from tqdm import tqdm

# if __name__ == '__main__':
#     model = Build_MultiModel_ShareBackbone()





#     label_path='/root/work2023/test_example/TRSL_ALL'




#     IMAGE_PATH = '/vepfs/gaowh/tr_eyesl/'

#     TRAIN_LISTS = ['train.csv']
#     TEST_LISTS = ['test.csv']
#     val_transforms = Image_Transforms(mode='val', dataloader_id=1, square_size=256, crop_size=224).get_transforms()






#     validate_loader = load_data_multi(label_path, TEST_LISTS, IMAGE_PATH, 16, val_transforms,'test')

#     val_bar = tqdm(validate_loader)
#     for val_data in val_bar:
#         val_images,val_labels,imgids = val_data


#         outputs = model(val_images)
#         print(outputs.shape)



#     # input = torch.randn(4, 40960)  # 
#     # segment_consensus = SegmentConsensus(40960, 256)
#     # output = segment_consensus(input)


#     # inputs = torch.randn(2, 3, 224, 224)
#     # output = model(inputs, inputs)
#     # print(output)


