import sys
import os
from argparse import ArgumentParser

# assert len(sys.argv) == 3, 'Args are wrong.'
#
# input_path = sys.argv[1]
# output_path = sys.argv[2]
#
# assert os.path.exists(input_path), 'Input model does not exist.'
# assert not os.path.exists(output_path), 'Output filename already exists.'
# assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
from share import *
from cldm.model import create_model

parser = ArgumentParser()
parser.add_argument("--SD_path", type=str, default="./models/v2-1_512-ema-pruned.ckpt")
parser.add_argument("--parsing_path", type=str, default="./face_parsing/res/cp/79999_iter.pth")
parser.add_argument("--output_path", type=str, default="./models/control_sd21_v13_ini.ckpt")
args = parser.parse_args()


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


model = create_model(config_path='models/cldm_v21_parsing14.yaml')

SD_pretrained_weights = torch.load(args.SD_path)
if 'state_dict' in SD_pretrained_weights:
    SD_pretrained_weights = SD_pretrained_weights['state_dict']

parsing_weights = torch.load(args.parsing_path)
if 'state_dict' in parsing_weights:
    parsing_weights = parsing_weights['state_dict']

scratch_dict = model.state_dict()

target_dict = {}

# TODO: arface weights' acquire, cldm_parsing-> tool_add_control_sd21

for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_')
    if is_control:
        copy_k = 'model.diffusion_' + name
    else:
        copy_k = k

    # 针对插入了lankmark_linear后，inputblocks层数的索引变化
    offset = copy_k.split(".")
    if len(offset)>4:
        if offset[2] == "input_blocks" and int(offset[4]) > 1:
            offset[4] = str(int(offset[4]) - 1)
        elif offset[2] == "middle_block" and int(offset[3]) > 1:
            offset[3] = str(int(offset[3]) - 1)
    offset_k = ".".join(offset)
    # ----

    if copy_k in SD_pretrained_weights:
        target_dict[k] = SD_pretrained_weights[copy_k].clone()
    elif offset_k in SD_pretrained_weights:
        target_dict[k] = SD_pretrained_weights[offset_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

#parsing 部分的权重
controlnet_parsing_weights = list(scratch_dict.keys())[-191:]
for k in range(len(controlnet_parsing_weights)):
    target_dict[controlnet_parsing_weights[k]] = parsing_weights[list(parsing_weights.keys())[k]]
    print(f'These weights can load : {controlnet_parsing_weights[k]}  <-  {list(parsing_weights.keys())[k]}')

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), args.output_path)
print('Done.')
