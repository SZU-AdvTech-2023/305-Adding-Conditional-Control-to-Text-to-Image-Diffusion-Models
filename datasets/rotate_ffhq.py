import copy
import random
import logging
import json
from typing import Sequence, Dict, Union
import cv2
import numpy
import torch
import numpy as np
import os
import PIL.Image as Image
import time
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import pickle
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale

logger = logging.getLogger(__name__)


class ffhq_dataset(data.Dataset):

    def __init__(
            self,
            file_list: str,
            out_size: int,
            crop_type: str,
    ) -> "ffhq_dataset":
        super(ffhq_dataset, self).__init__()
        self.file_list = file_list

        # rotate face part
        self.render_mask_path = os.path.join(self.file_list, "mask_face_new")
        self.coeff_path = os.path.join(self.file_list, "coeff_face_new")

        # front face part
        files = []
        taget_face_list = os.listdir(os.path.join(file_list, "ffhq_detection"))
        taget_face_list.remove("detections")
        for line in taget_face_list:
            path = line.strip()
            if path:
                angele = np.load(os.path.join(self.coeff_path, path.replace(".png", ".npy")))
                if -np.pi / 12 < angele.reshape(-1)[0] < np.pi / 12 and -np.pi / 12 < angele.reshape(-1)[1] < np.pi / 12 \
                        and -np.pi / 12 < angele.reshape(-1)[2] < np.pi / 12:
                    files.append(os.path.join(self.file_list+"/ffhq_detection", path))
        self.img_paths = files

        # parsing
        self.parsing_path = os.path.join(self.file_list, "parsing_face")

        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.

        gt_path = self.img_paths[index]
        img_name = gt_path.split("/")[-1]
        mask_path = os.path.join(self.render_mask_path, img_name)
        parsing_path = os.path.join(self.parsing_path, img_name)
        success = False
        for _ in range(3):
            try:
                frontface_img = Image.open(gt_path).convert("RGB")
                mask_face = Image.open(mask_path).convert("RGB")
                parsing_face = Image.open(parsing_path).convert('P')
                h, w, _ = np.array(frontface_img).shape
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"

        transform = transforms.Compose([])
        # resize
        if self.out_size != h or self.out_size != w:
            resize = transforms.Resize(self.out_size)
            transform.transforms.append(resize)

        transform.transforms.append(transforms.ToTensor())
        # 裁切
        if self.crop_type == "center":
            crop = transforms.CenterCrop(self.out_size)
            transform.transforms.append(crop)
        elif self.crop_type == "random":
            crop = transforms.RandomCrop(self.out_size)
            transform.transforms.append(crop)

        # img_gt = transform(frontface_img)
        # [0,1]
        img_gt = np.array(frontface_img, dtype=np.float32)
        img_gt = (img_gt / 255.0)

        # [-1, 1]
        target = (img_gt * 2 - 1)

        # arcface part
        # arcface transform
        trans = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        im_emb = trans(frontface_img)

        # mask part
        source = np.array(mask_face, dtype=np.float32)
        source = (source / 255.0)

        # parsing part
        parsing_gt = np.array(parsing_face).astype(np.int64)

        frontface_img.close()
        mask_face.close()
        return dict(jpg=target, txt="", hint=source, id_emb=im_emb, parsing_gt=parsing_gt)

    def __len__(self) -> int:
        return len(self.img_paths)
