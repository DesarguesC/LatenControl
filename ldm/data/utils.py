# -*- coding: utf-8 -*-

import cv2
import numpy as np
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_tensor
from transformers import CLIPProcessor
from basicsr.utils import img2tensor
import os, shutil
from ldm.util import resize_numpy_image as rs 


class PILtoTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        sample['jpg'] = to_tensor(sample['jpg'])
        if 'openpose' in sample:
            sample['openpose'] = to_tensor(sample['openpose'])
        return sample


class AddCannyFreezeThreshold(object):

    def __init__(self, low_threshold=100, high_threshold=200):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __call__(self, sample):
        # sample['jpg'] is PIL image
        x = sample['jpg']
        img = cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)
        canny = cv2.Canny(img, self.low_threshold, self.high_threshold)[..., None]
        sample['canny'] = img2tensor(canny, bgr2rgb=True, float32=True) / 255.
        return sample


class AddCannyRandomThreshold(object):

    def __init__(self, low_threshold=100, high_threshold=200, shift_range=50):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.threshold_prng = np.random.RandomState()
        self.shift_range = shift_range

    def __call__(self, sample):
        # sample['jpg'] is PIL image
        x = sample['jpg']
        img = cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)
        low_threshold = self.low_threshold + self.threshold_prng.randint(-self.shift_range, self.shift_range)
        high_threshold = self.high_threshold + self.threshold_prng.randint(-self.shift_range, self.shift_range)
        canny = cv2.Canny(img, low_threshold, high_threshold)[..., None]
        sample['canny'] = img2tensor(canny, bgr2rgb=True, float32=True) / 255.
        return sample


class AddStyle(object):

    def __init__(self, version):
        self.processor = CLIPProcessor.from_pretrained(version)
        self.pil_to_tensor = transforms.ToTensor()

    def __call__(self, sample):
        # sample['jpg'] is PIL image
        x = sample['jpg']
        style = self.processor(images=x, return_tensors="pt")['pixel_values'][0]
        sample['style'] = style
        return sample


class AddSpatialPalette(object):

    def __init__(self, downscale_factor=64):
        self.downscale_factor = downscale_factor

    def __call__(self, sample):
        # sample['jpg'] is PIL image
        x = sample['jpg']
        img = cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        color = cv2.resize(img, (w // self.downscale_factor, h // self.downscale_factor), interpolation=cv2.INTER_CUBIC)
        color = cv2.resize(color, (w, h), interpolation=cv2.INTER_NEAREST)
        sample['color'] = img2tensor(color, bgr2rgb=True, float32=True) / 255.
        return sample


def get_bit(num: int) -> int:
    if num == 0:
        return 1
    assert num > 0
    cnt = 0
    while num != 0:
        cnt += 1
        num = num // 10
    return cnt

def get_name(cnt_base: int) -> str:
    cc = 10
    cn = get_bit(cnt_base)
    return '0' * (cc - cn) + str(cnt_base)

def move(coco_path: str, target_path: str, base_num: int, resize=False, max_resolution=512*512):
    # TODO: MOVE every images into training base path

    # base_num: responds to gpu (total gpu ammount)

    tot_cnt = 0
    assert os.path.exists(coco_path), 'invalid coco_path'

    coco_path = coco_path if coco_path.endswith('/') else coco_path + '/'
    target_path = target_path if target_path.endswith('/') else target_path + '/'
    
    # print('coco_path: ', coco_path)
    # print('target_path', target_path)


    # TODO: create save direction
    ori_img_path = target_path + 'ori_img/'
    point_img_path = target_path + 'point_img/'
    prompt_path = target_path + 'prompts/'

    if not os.path.exists(ori_img_path):
        os.mkdir(ori_img_path)
    if not os.path.exists(point_img_path):
        os.mkdir(point_img_path)
    if not os.path.exists(prompt_path):
        os.mkdir(prompt_path)

    prompt_file = open(prompt_path + 'prompts.txt', 'w')

    for i in range(base_num):
        # TODO: move ori image
        ori, poi, pro = coco_path + f'ori_img-{i}/', coco_path + f'point_img-{i}/', coco_path + f'prompts-{i}/'

        if not os.path.exists(ori):
            break
        assert os.path.exists(ori)
        assert os.path.exists(poi)
        assert os.path.exists(pro)

        ori_list = os.listdir(ori)
        poi_list = os.listdir(poi)
        
        with open(pro + 'prompts.txt', 'r') as f:
            prompts = f.readlines()
        
        assert len(ori_list) == len(poi_list), f'data length unequal Erro, ori_list: {len(ori_list)}, poi_list: {len(poi_list)} at {i}'
        assert len(ori_list) == len(prompts), 'data length unequal Error'

        for k in range(len(ori_list)):
            ori_image = ori + ori_list[k]
            poi_image = poi + poi_list[k]
            prompt = prompts[k]

            name = get_name(tot_cnt)

            # shutil.copyfile(ori_image, ori_img_path + name + '.jpg')
            # shutil.copyfile(poi_image, point_img_path + name + '.jpg')
            
            if resize:
                try:
                    original = cv2.imread(ori_img_path + name + '.jpg')
                    point = cv2.imread(point_img_path + name + '.jpg')
                    original = rs(original, max_resolution=max_resolution)
                    point = rs(point, max_resolution=max_resolution)
                    cv2.imwrite(ori_img_path + name + '.jpg', original)
                    cv2.imwrite(point_img_path + name + '.jpg', point)
                except:
                    print(ori_img_path + name + '.jpg', point_img_path + name + '.jpg')
                    
                    
            prompt_file.write(prompt)
            if not prompt.endswith('\n'):
                prompt_file.write('\n')

            tot_cnt += 1

"""

target_path
    | --- ori_img   --- name.jpg ...
    | --- point_img --- name.jpg ...
    | --- prompts   --- prompts.txt

"""
