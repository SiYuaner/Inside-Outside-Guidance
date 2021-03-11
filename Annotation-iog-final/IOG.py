import scipy.misc as sm
import glob
import numpy as np
import os
import cv2
import torch
from torch.nn.functional import upsample
from dataloaders.helpers import *
from modeling.IOG import *
from dataloaders.helpers import *

def get_distancemap(sigma, elem, elem_cp,pad_pixel):
    _target = elem
    _cp = elem_cp
    targetshape=_target.shape
    if np.max(_target) == 0:
        distancemap = np.zeros([targetshape[0],targetshape[1],2], dtype=_target.dtype) #  TODO: handle one_mask_per_point case
    else:
        _points = GetDistanceMap_user(_target, _cp,pad_pixel)        
        distancemap = make_gt(_target, _points, sigma=sigma, one_mask_per_point=False)
    custom_max=255.
    tmp = distancemap
    tmp = custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
    return tmp

def totensor(tmp):
    if tmp.ndim == 2:
        tmp = tmp[:, :, np.newaxis]
    tmp = tmp.transpose((2, 0, 1))
    tmp = tmp[np.newaxis, :, :]
    tmp= torch.from_numpy(tmp)
    return tmp
    
def getbg(bgx,bgy,bgyw,bgyh,w,h):
    _bg = np.zeros((w,h)) 
    _bg[bgx:bgx+bgyw,bgy:bgy+bgyh] = 1
    _bg = _bg.astype(np.float32)   
    return _bg
    
def getcp(cx,cy,w,h):
    _cp = np.zeros((w,h)) 
    _cp[:cy,:cx] = 1
    _cp = _cp.astype(np.float32)  
    return _cp

def loadnetwork():
    # Set gpu_id to -1 to run in CPU mode, otherwise set the id of the corresponding gpu
    gpu_id = 0
    device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('Initial the IOG using GPU: {} '.format(gpu_id))
    
    # Setting parameters
    resume_epoch = 80  # Default is 0, change if want to resume
    nInputChannels = 5  # Number of input channels (RGB + heatmap of extreme points)
 
    # Results and model directories (a new directory is generated for every run)
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    if resume_epoch == 0:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    else:
        run_id = 0
    save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))
    if not os.path.exists(os.path.join(save_dir, 'models')):
        os.makedirs(os.path.join(save_dir, 'models'))    
    # Network definition
    modelName = 'IOG_pascal'
    net = IOG(nInputChannels=nInputChannels,num_classes=1,
                            backbone='resnet101',
                            output_stride=16,
                            freeze_bn=False)    
    #load models
    pretrain_dict=torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'))
    net_dict=net.state_dict()
    for k, v in pretrain_dict.items():  
        if k in net_dict:      
            net_dict[k] = v
        else:
            print('skil parameters:',k)

    net.load_state_dict(net_dict)
    net.to(device)
    net.eval()
    print('end')
    return net
    
def IOG_getmask(bgpoint,cppoint,image,net): 
    
    with torch.no_grad():
        device = 'cuda'
        cx = cppoint[0]
        cy = cppoint[1]
        bgx = bgpoint[0]
        bgy = bgpoint[1]
        bgyw = bgpoint[2]-bgx
        bgyh = bgpoint[3]-bgy
        w,h,channel =image.shape
        bg = getbg(bgy,bgx,bgyh,bgyw,w,h)
        cp = getcp(cx,cy,w,h)    
        crop_image = crop_from_mask(image, bg, relax=30, zero_pad=True)
        crop_bg = crop_from_mask(bg, bg, relax=30, zero_pad=True)
        crop_cp = crop_from_mask(cp, bg, relax=30, zero_pad=True)
        crop_image = fixed_resize(crop_image, (512,512) )
        crop_bg = fixed_resize(crop_bg, (512,512) )
        crop_cp = fixed_resize(crop_cp, (512,512) )
        distancemap =  get_distancemap(sigma=10,elem=crop_bg,elem_cp=crop_cp,pad_pixel=10)    
        distancemap = totensor(distancemap)
        crop_image = totensor(crop_image)
        distancemap = distancemap.float()
        crop_image = crop_image.float()
        
        inputs=torch.cat([crop_image,distancemap],1)                
        inputs = inputs.to(device)
        glo1,glo2,glo3,glo4,refine= net.forward(inputs)  
        output_refine = upsample(refine, size=(512, 512), mode='bilinear', align_corners=True)
        
        #generate result
        jj=0  
        outputs = output_refine.to(torch.device('cpu'))
        pred = np.transpose(outputs.data.numpy()[jj, :, :, :], (1, 2, 0))
        pred = 1 / (1 + np.exp(-pred))
        pred = np.squeeze(pred)
        gt = bg 
        bbox = get_bbox(gt, pad=30, zero_pad=True)
        result = crop2fullmask(pred, bbox, gt, zero_pad=True, relax=0,mask_relax=False)
        # 0~1
        resultmax,resultmin = result.max(),result.min()
        result = (result-resultmin)/(resultmax-resultmin)
        result = (result>0.3)*255
        cp = cp.astype(np.uint8)
#        sm.imsave('result.png' , result)
    return result  
    
    #    #show distance map
    #    jj=0
    #    outputs = crop_image.to(torch.device('cpu'))
    #    pred_img = np.transpose(outputs.data.numpy()[jj, :, :, :], (1, 2, 0))   
    #    outputs = distancemap.to(torch.device('cpu'))
    #    pred = np.transpose(outputs.data.numpy()[jj, :, :, :], (1, 2, 0))
    #    pred_gt = np.squeeze(pred)
    #    pred_img[:,:,2] = pred_img[:,:,2] + pred_gt[:,:,0]*200
    #    pred_img[:,:,0] = pred_img[:,:,0] + pred_gt[:,:,1]*200
    #    sm.imsave(os.path.join(save_dir_res_list[0],'distance.png' ),pred_img )
