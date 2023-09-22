import json
import cv2
import os
from basicsr.utils import img2tensor
from ldm.data.utils import move
import cv2

Inter = {
    
}

def Resize(img_path: str, shape: tuple):
    img = cv2.read()


class PointDataset():
    def __init__(self, base_path: str, base_num: int, save_path=None, max_resolution=512*512):
        # base_path: coco_path -> 'COCO'
        self.max_resolution = max_resolution
        
        base_path = base_path if base_path.endswith('/') else base_path + '/'
        if save_path is not None:
            save_path = save_path if save_path.endswith('/') else save_path + '/'
        save_path = base_path  + 'total_data/' if save_path == None else save_path

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        move(base_path, save_path, base_num, resize=True, max_resolution=self.max_resolution)

        ori_img = save_path + 'ori_img/'
        point_img = save_path + 'point_img/'
        
        self.path = [ori_img, point_img]

        prompt = save_path + 'prompts/prompts.txt'
        prompts = open(prompt, 'r').readlines()

        ori_list, point_list = os.listdir(ori_img), os.listdir(point_img)
        assert len(ori_list) == len(point_list), f'length unequal error-1, len(ori_list) = {len(ori_list)}, len(point_list) = {len(point_list)}'
        assert len(ori_list) == len(prompts), f'length unequal error-2, len(ori_list) = {len(ori_list)}, len(prompts) = {len(prompts)}'

        self.ori_imgs = ori_list
        self.point_imgs = point_list
        self.prompts = prompts
        
        H, W, _ = cv2.imread(ori_img + self.ori_imgs[0]).shape
        self.item_shape = (H, W)

    def __getitem__(self, idx):
        ori_img = cv2.imread(self.path[0] + self.ori_imgs[idx])
        ori_img = img2tensor(ori_img, bgr2rgb=True, float32=True) / 255.

        point_img = cv2.imread(self.path[1] + self.point_imgs[idx])
        point_img = img2tensor(point_img, bgr2rgb=True, float32=True) / 255.
        
        # assert ori_img.shape == point_img.shape, f'ori_image.shape = {ori_img.shape}, point_img.shape = {point_img.shape}'
        # need ?

        prompt = self.prompts[idx]

        return {
            'prompt': prompt,
            'ori_img': ori_img,
            'poi_img': point_img
        }

    def __len__(self):
        return len(self.ori_imgs)
