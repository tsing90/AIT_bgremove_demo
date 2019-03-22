
import time
import torch
import argparse
import os
import math, tqdm
import cv2
import numpy as np

from dplab.model import DeepLabModel
from model import network
import gen_trimap

def resize_pad(img, def_size):
    h, w = img.shape[:2]
    if max(h, w) > def_size or h % 16 != 0 or w % 16 != 0:
        new_h, new_w = h, w
        if max(h, w) > def_size:
            ratio = def_size / max(h, w)
            new_h, new_w = int(ratio * new_h), int(ratio * new_w)
        if new_h % 16 != 0 or new_w % 16 != 0:
            new_h, new_w = math.ceil(new_h / 16) * 16, math.ceil(new_w / 16) * 16

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # padding if w!=h
    if img.shape[0] != img.shape[1]:
        max_size = max(img.shape[0], img.shape[1])
        pad_input = True

        pad_h = int(max_size - img.shape[0])
        pad_w = int(max_size - img.shape[1])
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

    else:
        pad_h, pad_w = None, None

    assert def_size, def_size == img.shape[:2]
    return img, (pad_h, pad_w)

out_path='static/download/'
trimap_size = 512
alpha_size = 1024
without_gpu = False

def mat_1(img_path, object=False):
    
    assert os.path.isfile(img_path), 'Wrong image path:{}'.format(img_path)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    ## trimap stage
    model_t = DeepLabModel('xception_model')
    print('deeplabv3 model loaded !')

    img_t, pad_t = resize_pad(img, trimap_size)
    assert trimap_size <513
    #img_t = img_t /127.5 - 1
    img_t = img_t[:,:,::-1]
    labels = model_t.run(img_t)

    if object:
        obj_list = [2,3,5,8,9,11,12,13,14,15,18]
        for idx in obj_list:
            labels[labels == idx] = 255
        labels[labels < 250] = 0
    else:
        labels[labels == 15] = 255
        labels[labels < 250] = 0
    labels = labels.astype(np.uint8)

    return labels

def matting(img_path, object=False):
    assert os.path.isfile(img_path), 'Wrong image path:{}'.format(img_path)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    ## trimap stage
    model_t = DeepLabModel('xception_model')
    print('deeplabv3 model loaded !')

    img_t, pad_t = resize_pad(img, trimap_size)
    assert trimap_size < 513
    # img_t = img_t /127.5 - 1
    img_t = img_t[:, :, ::-1]
    labels = model_t.run(img_t)

    if object:
        obj_list = [2, 3, 5, 8, 9, 11, 12, 13, 14, 15, 18]
        for idx in obj_list:
            labels[labels == idx] = 255
        labels[labels < 250] = 0
    else:
        labels[labels == 15] = 255
        labels[labels < 250] = 0

    alpha = labels.astype(np.uint8)

    # fusion stage
    if pad_t[0]:
        alpha = alpha[:-pad_t[0]]
    elif pad_t[1]:
        alpha = alpha[:, :-pad_t[1]]
    alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_CUBIC)
    new = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    new[:,:,3] = alpha
    new_name = os.path.join(out_path, os.path.splitext(os.path.basename(img_path))[0])
    cv2.imwrite(new_name + '-AIT.png', new)

    return alpha, new

def mat_2(img_path, labels):

    # alpha stage
    if without_gpu:
        print("use CPU !")
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
            print("use GPU !")
            device = torch.device('cuda:0,1')

    torch.set_grad_enabled(False)

    def load_model():
        model_path = os.path.join('ckpt/pre_train_m_net/model/model_obj.pth')
        assert os.path.isfile(model_path), 'Wrong model path: {}'.format(model_path)
        if without_gpu:
            myModel = torch.load(model_path, map_location=lambda storage, loc: storage)
        else:
            myModel = torch.load(model_path)
        myModel.eval()
        myModel.to(device)
        return myModel

    def seg_process(inputs, net):
        alpha = net(inputs[0], inputs[1])
        if without_gpu:
            alpha = alpha.data.numpy()
        else:
            alpha = alpha.cpu().data.numpy()

        alpha = alpha[0][0] * 255.0
        #alpha = alpha.astype(np.uint8)
        return alpha

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    m_net = load_model()

    trimap_src = gen_trimap.get_trimap(labels)
    img_a, pad_a = resize_pad(img, alpha_size)
    trimap_src = cv2.resize(trimap_src, (alpha_size, alpha_size), interpolation=cv2.INTER_NEAREST)


    img_a = img_a / 255.0
    trimap = trimap_src.copy()
    trimap[trimap == 0] = 0
    trimap[trimap == 128] = 1
    trimap[trimap == 255] = 2
    trimap = np.eye(3)[trimap.reshape(-1)].reshape(list(trimap.shape) + [3])

    tensor_img = torch.from_numpy(img_a.astype(np.float32)[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
    tensor_tri = torch.from_numpy(trimap.astype(np.float32)[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
    tensor_img = tensor_img.to(device)
    tensor_tri = tensor_tri.to(device)

    alpha = seg_process((tensor_img, tensor_tri), m_net)

    # fusion stage
    if pad_a[0]:
        alpha = alpha[:-pad_a[0]]
    else:
        alpha = alpha[:, :-pad_a[1]]
    alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_CUBIC)
    new = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    new[:,:,3] = alpha
    new_name = os.path.join(out_path, os.path.splitext(os.path.basename(img_path))[0])
    cv2.imwrite(new_name + '-AIT.png', new)

    #cv2.imwrite(new_name + '-lb.png', trimap_src)
