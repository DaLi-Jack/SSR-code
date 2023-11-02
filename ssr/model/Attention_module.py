import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention_RoI_Module(nn.Module):
    def __init__(self,img_feat_channel=256,global_dim=256,global_detach=True):
        super(Attention_RoI_Module,self).__init__()
        self.global_detach = global_detach
        self.channel_mlp = nn.Sequential(
            nn.Linear(in_features=global_dim, out_features=img_feat_channel),
            nn.ReLU(),
            nn.Linear(in_features=img_feat_channel, out_features=img_feat_channel),
            nn.ReLU(),
            nn.Linear(in_features=img_feat_channel, out_features=img_feat_channel),
            nn.Sigmoid(),
        )
        self.post_conv=nn.Sequential(
            nn.Conv2d(in_channels=img_feat_channel, out_channels=img_feat_channel, kernel_size=5,padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=img_feat_channel, out_channels=img_feat_channel, kernel_size=5,padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=img_feat_channel, out_channels=img_feat_channel, kernel_size=5,padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=img_feat_channel, out_channels=img_feat_channel, kernel_size=5,padding=2),
        )
        self.pre_conv=nn.Sequential(
            nn.Conv2d(in_channels=img_feat_channel, out_channels=img_feat_channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=img_feat_channel, out_channels=img_feat_channel, kernel_size=1),
        )


    def forward(self,img_feat,global_feat,bdb_grid):
        roi_feat = self.pre_conv(img_feat)              # keep the same size
        if self.global_detach:
            global_feat=global_feat.detach()
        channel_wise_weight=self.channel_mlp(global_feat)
        out_feat=channel_wise_weight.unsqueeze(2).unsqueeze(3)*roi_feat
        out_feat=self.post_conv(out_feat)+roi_feat
        ret_dict={
            "roi_feat":out_feat,
        }
        ret_dict['channel_atten_weight']=channel_wise_weight
        return ret_dict
