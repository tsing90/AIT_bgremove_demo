
import time
import cv2
import torch
import argparse
import numpy as np
import os
import math, tqdm
import torch.nn.functional as F

from model import network

# load model
def load_model():
    t_path = os.path.join('./ckpt', 'pre_train_t_net', 'model/ckpt_lastest.pth')
    m_path = os.path.join('./ckpt', 'pre_train_m_net', 'model/ckpt_lastest.pth')
    assert os.path.isfile(t_path), 'Wrong model path: {}'.format(m_path)
    print('Loading model ...')

    t_model = network.net_T()
    m_model = network.net_M()
    t_model.load_state_dict(torch.load(t_path, map_location=lambda storage, loc: storage)['state_dict'])
    m_model.load_state_dict(torch.load(m_path, map_location=lambda storage, loc: storage)['state_dict'])

    t_model.eval()
    m_model.eval()
    t_model.to(device)
    m_model.to(device)

    return t_model, m_model

#################################
# resize input images
def resize_pad(img, def_size):
    h, w = img.shape[:2]
    if max(h, w) > def_size or h % 16 != 0 or w % 16 != 0:
        new_h, new_w = h, w
        if max(h, w) > def_size:
            ratio = def_size / max(h, w)
            new_h, new_w = int(ratio * new_h), int(ratio * new_w)
        if new_h % 16 != 0 or new_w % 16 != 0:
            new_h, new_w = math.ceil(new_h / 16) * 16, math.ceil(new_w / 16) * 16

        img = cv2.resize(img, (new_w, new_h))

    # padding if w!=h
    if img.shape[0] != img.shape[1]:
        max_size = max(img.shape[0], img.shape[1])
        pad_h = int(max_size - img.shape[0])
        pad_w = int(max_size - img.shape[1])
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

    else:
        max_size = img.shape[0]
        pad_h, pad_w = None, None

    return img, max_size, (pad_h, pad_w)


def seg_process(img_t, img_m, re_size, net_t, net_m):
    #print('t_net input shape:', img_t.shape)
    trimap = net_t(img_t)

    trimap = torch.argmax(trimap[0], dim=0)
    trimap = trimap.data.numpy().astype(np.uint8)
    trimap = cv2.resize(trimap, (re_size, re_size), interpolation=cv2.INTER_NEAREST)
    trimap = np.eye(3)[trimap.reshape(-1)].reshape(list(trimap.shape) + [3])
    trimap = torch.from_numpy(trimap.astype(np.float32)[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
    trimap = trimap.to(device)
    """
    trimap = torch.argmax(trimap, dim=1)
    #trimap = trimap.long()
    trimap = torch.eye(3)[trimap.reshape(-1)].reshape([1,3]+list(trimap.shape[1:]))  # created in cpu!
    trimap = F.interpolate(trimap, (re_size, re_size))
    trimap = trimap.to(device)
    """
    #print('size:',re_size, 'shape:',img_m.shape, trimap.shape)
    assert img_m.shape == trimap.shape
    alpha = net_m(img_m, trimap)
    alpha = alpha.data.numpy() 

    alpha = alpha[0][0] * 255.0
    alpha = alpha.astype(np.uint8)
        
    return alpha

out_path = 'static/download/'
trimap_size = 512
alpha_size = 800

torch.set_num_threads(16)
torch.set_grad_enabled(False)
print("use CPU !")
device = torch.device('cpu')

t_model, m_model = load_model()

def matting(img_path):
    assert os.path.isfile(img_path), 'Wrong image path:{}'.format(img_path)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    img_src = img.copy()
    
    img, _, _ = resize_pad(img_src, trimap_size)
    img_m, re_size, pad = resize_pad(img_src, alpha_size)
    img = img / 255.0
    img_m = img_m / 255.0
    tensor_img = torch.from_numpy(img.astype(np.float32)[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
    tensor_img_m = tensor_img = torch.from_numpy(img_m.astype(np.float32)[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
    tensor_img = tensor_img.to(device)
    tensor_img_m = tensor_img_m.to(device)

    #print('before tensor:',img_m.shape, re_size)
    start = time.time()
    frame_seg = seg_process(tensor_img, tensor_img_m, re_size, t_model, m_model)
    print('time taken: {:05f} s'.format(time.time() - start))

    # show a frame
    if pad[0]:
        frame_seg = frame_seg[:-pad[0]]
    elif pad[1]:
        frame_seg = frame_seg[:, :-pad[1]]

    frame_seg = cv2.resize(frame_seg, (w, h), interpolation=cv2.INTER_CUBIC)

    #cv2.imwrite('1-alpha.png', frame_seg)

    img_src[frame_seg == 0, :3] = 0
    new = cv2.cvtColor(img_src, cv2.COLOR_BGR2BGRA)
    new[:,:,3] = frame_seg
    new_name = os.path.join(out_path, os.path.splitext(os.path.basename(img_path))[0])
    cv2.imwrite(new_name + '-AIT.png', new)

