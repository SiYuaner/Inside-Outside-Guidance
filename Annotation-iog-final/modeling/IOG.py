import torch
import torch.nn as nn
import torch.nn.functional as F

#network
from .globalNet import globalNet
from .refineNet import refineNet
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.backbone import build_backbone

affine_par = True
class PSPModule(nn.Module):
    """
    Pyramid Scene Parsing module
    """
    def __init__(self, in_features=2048, out_features=512, sizes=(1, 2, 3, 6), n_classes=1):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage_1(in_features, size) for size in sizes])
        self.bottleneck = self._make_stage_2(in_features * (len(sizes)//4 + 1), out_features)
        self.relu = nn.ReLU()

    def _make_stage_1(self, in_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_features, in_features//4, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(in_features//4, affine=affine_par)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def _make_stage_2(self, in_features, out_features):
        conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_features, affine=affine_par)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages]
        priors.append(feats)
        bottle = self.relu(self.bottleneck(torch.cat(priors, 1)))#512 32 32
        return bottle

class IOG(nn.Module):
    """
    main network for iog
    """
    def __init__(self, backbone='resnet',output_stride=16, num_classes=1,nInputChannels=5,freeze_bn=False):
        super(IOG, self).__init__()
        output_shape = 128
        channel_settings = [512, 1024, 512, 256]
        BatchNorm = nn.BatchNorm2d
        self.global_net = globalNet(channel_settings, output_shape, num_classes)
        self.refine_net = refineNet(channel_settings[-1], output_shape, num_classes)        
        self.backbone = build_backbone(backbone, output_stride, BatchNorm,nInputChannels)
        self.psp4 = PSPModule(in_features=2048, out_features=512, sizes=(1, 2, 3, 6), n_classes=256)
        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        low_level_feat_4, low_level_feat_3,low_level_feat_2,low_level_feat_1 = self.backbone(input)
        low_level_feat_4 = self.psp4(low_level_feat_4)   
        res_out = [low_level_feat_4, low_level_feat_3,low_level_feat_2,low_level_feat_1]    
        global_fms, global_outs = self.global_net(res_out)       
        refine_out = self.refine_net(global_fms)
        return global_outs[0],global_outs[1],global_outs[2],global_outs[3],refine_out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.global_net,self.refine_net,self.psp4]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
                            


class IOG_loop(nn.Module):
    def __init__(self, backbone='resnet',output_stride=16, num_classes=1,nInputChannels=5,freeze_bn=False):
        super(IOG_loop, self).__init__()
        output_shape = 128
        BatchNorm = nn.BatchNorm2d        
        channel_settings = [512, 1024, 512, 256]#[2048, 1024, 512, 256]
        self.global_net = globalNet(channel_settings, output_shape, num_classes)
        self.refine_net = refineNet(channel_settings[-1], output_shape, num_classes)        
        self.backbone = build_backbone(backbone, output_stride, BatchNorm,nInputChannels)
        self.psp4 = PSPModule(in_features=2048+64, out_features=512, sizes=(1, 2, 3, 6))      
        self.ex_points = nn.Sequential(nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),

                                       nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),

                                       nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),


                                       nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),

                                       nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU())
        if freeze_bn:
            self.freeze_bn()

    def forward(self, input,distance_map):
        low_level_feat_4_ori, low_level_feat_3_ori,low_level_feat_2_ori,low_level_feat_1_ori = self.backbone(input)        
        distance_map = self.ex_points(distance_map)               
        feats_concat=torch.cat((low_level_feat_4_ori,distance_map),dim=1)#2048+64
        low_level_feat_4 = self.psp4(feats_concat)   
        res_out = [low_level_feat_4, low_level_feat_3_ori,low_level_feat_2_ori,low_level_feat_1_ori]       
        global_fms, global_outs = self.global_net(res_out)        
        refine_out = self.refine_net(global_fms)

        return refine_out
        
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.global_net,self.refine_net,self.psp4,self.ex_points]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p