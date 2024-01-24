import pickle
import config
import matplotlib.pyplot as plt
import PIL.Image as Image
import cv2
import einops
# import gradio as gr
import numpy as np
import torch
import random
import lmdb
from io import BytesIO
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from arcface_torch.RecogNetWrapper import RecogNetWrapper_50, RecogNetWrapper


device = torch.device("cuda:2")

model = create_model('./models/cldm_v21_parsing14.yaml').cpu()
model.load_state_dict(load_state_dict('./models/model_v14_after_epoch.ckpt', location="cuda:2"))
model.to(device)
ddim_sampler = DDIMSampler(model)

img_index = 495 #204 48 4 424 501
input_image = Image.open("/home/zhongtao/datasets/ffhq_rotate/ffhq_detection/"+f"{img_index:0{5}}"+".png").convert("RGB")

# control
mask_face = Image.open("/home/zhongtao/datasets/ffhq_rotate/test_face/"+f"{img_index:0{5}}"+".png").convert("RGB")
mask_trans = transforms.ToTensor()
control = mask_trans(mask_face).unsqueeze(0).to(device)
# control = model.apply_condition_encoder(control)

## embedding
processor = model.control_processor
trans = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])
source = trans(input_image)
source = source.unsqueeze(0).to(device)
processor.eval()
id_embed, total_embed = processor(source)
# id_embed, total_embed = processor(source)
# id_embed.reshape(1,-1,id_embed.shape[-1])

# seed = random.randint(0, 65535)
for i in range(2054,65535): #2051 to start
    seed = i
    seed_everything(seed)

    prompt = ""
    image_resolution = 512
    strength = 1.0
    guess_mode = True
    ddim_steps = 50
    scale = 6.0
    eta = 0.0
    # a_prompt ='best quality, extremely detailed, real scene'
    n_prompt = ""
    a_prompt = ""
    #n_prompt = ""
    num_samples = 1

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], "id_emb":[total_embed]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)], "id_emb":[torch.randn_like(total_embed)]}
    H, W, C = np.array(input_image).shape
    shape = (4, H // 8, W // 8)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                 shape, cond, verbose=False, eta=eta,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=un_cond)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(
        np.uint8)

    plt.subplot(1,2,1)
    plt.imshow(x_samples[0])
    plt.subplot(1,2,2)
    plt.imshow(input_image)
    mid_emb = processor(trans(Image.fromarray(x_samples[0])).unsqueeze(0).to(device))[0]
    id_similarity = float(torch.cosine_similarity(mid_emb, id_embed).cpu())
    plt.title("seed: "+ str(seed)+" "+"img_index: "+str(img_index)+" "+"id_similiarity:{:.6f}".format(id_similarity))
    plt.show()


