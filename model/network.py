
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
import numpy as np

from model.M_Net import M_net
from model.T_Net_psp import PSPNet

class net_T(nn.Module):
    # Train T_net
    def __init__(self):

        super(net_T, self).__init__()

        self.t_net = PSPNet()

    def forward(self, input):

    	# trimap
        trimap = self.t_net(input)
        return trimap

class net_M(nn.Module):
    # train M_net

    def __init__(self):

        super(net_M, self).__init__()
        self.m_net = M_net()

    def forward(self, input, trimap):

        # paper: bs, fs, us
        bg, unsure, fg = torch.split(trimap, 1, dim=1)

        # concat input and trimap
        m_net_input = torch.cat((input, trimap), 1)

        # matting
        alpha_r = self.m_net(m_net_input)
        # fusion module
        # paper : alpha_p = fs + us * alpha_r
        alpha_p = fg + unsure * alpha_r

        return alpha_p

class net_F(nn.Module):
    # end to end net *training*

    def __init__(self):

        super(net_F, self).__init__()

        self.t_net = PSPNet()
        self.m_net = M_net()

    def forward(self, t_img, m_img, crop_xy, patch_size):
    	# trimap shape: B, 3, H, W; value range after softmax: [0,1]
        trimap = self.t_net(t_img)
        trimap_softmax = F.softmax(trimap, dim=1)

        # crop&resize trimap into crop_size: data shape of: B, C, H, W
        x_list, y_list, crop_size_list = crop_xy
        b_size, c_size, h_old, w_old = trimap_softmax.shape
        tri_list = []
        for idx in range(b_size):
            x, y, crop_size = x_list[idx], y_list[idx], crop_size_list[idx]
            trimap_i = trimap_softmax[idx:idx+1]

            cropped = trimap_i[:, :, y:y+crop_size, x:x+crop_size].clone()
            _, _, h_crop, w_crop = cropped.shape
            diff_h, diff_w = crop_size - h_crop, crop_size - w_crop
            new_trimap = F.pad(cropped, (0, diff_w, 0, diff_h), "constant", 0)
            # resize trimap to patch_size
            new_trimap = F.interpolate(new_trimap, (patch_size, patch_size))  # default mode: 'nearest'

            tri_list.append(new_trimap)

        trimap_m = torch.cat(tri_list, 0)
        assert (b_size, c_size, patch_size, patch_size) == trimap_m.shape, \
            'trimap_softmax has wrong shape:{}'.format(trimap_m.shape)

        # paper: bs, fs, us
        bg, unsure, fg = torch.split(trimap_m, 1, dim=1)

        # concat input and trimap
        m_net_input = torch.cat((m_img, trimap_m), 1)

        # matting
        alpha_r = self.m_net(m_net_input)
        # fusion module
        # paper : alpha_p = fs + us * alpha_r
        alpha_p = fg + unsure * alpha_r

        #return trimap, alpha_p
        return trimap, alpha_p, trimap_m

class net_F_test(nn.Module):
    # end to end net *test/inference8

    def __init__(self):

        super(net_F, self).__init__()

        self.t_net = PSPNet()
        self.m_net = M_net()

    def forward(self, input):

        # trimap
        trimap = self.t_net(input)
        #trimap_softmax = F.softmax(trimap, dim=1)
        trimap_arg = torch.argmax(trimap, dim=1)  #shape: b, h, w
        trimap_softmax = torch.eye(3)[trimap_arg.reshape(-1)].reshape(trimap.shape)

        # paper: bs, fs, us
        bg, unsure, fg = torch.split(trimap_softmax, 1, dim=1)

        # concat input and trimap
        m_net_input = torch.cat((input, trimap_softmax), 1)

        # matting
        alpha_r = self.m_net(m_net_input)
        # fusion module
        # paper : alpha_p = fs + us * alpha_r
        alpha_p = fg + unsure * alpha_r

        return trimap, alpha_p
