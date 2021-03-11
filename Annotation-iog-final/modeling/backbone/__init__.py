from modeling.backbone import resnet

def build_backbone(backbone, output_stride, BatchNorm,nInputChannels):
    if backbone == 'resnet101':
        return resnet.ResNet101(output_stride, BatchNorm,nInputChannels=nInputChannels)
    elif backbone == 'resnet50':
        return resnet.ResNet50(output_stride, BatchNorm,nInputChannels=nInputChannels)    
    else:
        raise NotImplementedError
