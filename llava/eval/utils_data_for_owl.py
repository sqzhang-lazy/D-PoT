from torch.utils.data import Dataset
import torch
import pickle
from tqdm import tqdm
import action_type
import numpy as np
import jax.numpy as jnp
import random
import argparse, jsonlines, os
from PIL import Image
import json

img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 2048),
    "detr": (100, 256),
    "vit": (577, 768),
    "vit-large": (145, 1024),
    "vit-global": (1, 768),
    "vit-merge": (578, 768),
}

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_root', type=str, default='dataset/owl/general_parsed_episode_owl')
#     parser.add_argument('--output_dir', type=str, default='experiments')
#     parser.add_argument('--model', type=str, default='google/flan-t5-base')
#     parser.add_argument('--data_ratio', type=float, default=None)
#     parser.add_argument('--eval_name', type=str, default=None, help='the saved subset name used for evaluation')
#     parser.add_argument('--local_rank', type=int, default=-1)
#     parser.add_argument('--epoch', type=int, default=20)
#     parser.add_argument('--lr', type=float, default=5e-5)
#     parser.add_argument('--warmup_ratio', type=float, default=0.1)
#     parser.add_argument('--bs', type=int, default=2)
#     parser.add_argument('--debug_num', type=int, default=2)
#     parser.add_argument('--input_len', type=int, default=512)
#     parser.add_argument('--output_len', type=int, default=128)
#     parser.add_argument('--img_dim', type=int, default=512)
#     parser.add_argument('--eval_bs', type=int, default=16)
#     parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
#     parser.add_argument('--all_data', type=float, default=None, help='whether using all the data for training. Set the ratio for google apps to save computation')
#     parser.add_argument('--eval_subset', type=str, default=None, help='use which subset for evaluation/test when training with all data')
#     parser.add_argument('--use_history', type=int, default=None, help='only evaluate the model at the final epoch')
#     parser.add_argument('--use_img_history', action='store_true', help='only evaluate the model at the final epoch')
#     parser.add_argument('--use_layout', action='store_true', help='only evaluate the model at the final epoch')
#     parser.add_argument('--transform_axis', action='store_true', help='only for baseline to improve inference speed')
#     parser.add_argument('--use_generate', default=True, action='store_true', help='only for baseline to improve inference speed')
#     parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
#     parser.add_argument('--user_msg', type=str, default="debug", help='experiment type in the save_dir')
#     parser.add_argument('--img_type', type=str, default="clip", choices=['detr', 'clip', 'blip','vit','vit-large','vit-global','vit-merge'], help='type of image features')
#     parser.add_argument('--evaluate_dir', type=str, default=None, help='the directory of model for evaluation')
#     parser.add_argument('--seed', type=int, default=42, help='random seed')

#     args = parser.parse_args()
#     return args

def load_data(args, split):
    target_text = []
    source_text = []
    source_image = []


    with open("data/general_test_llava_woplan.json", 'r') as rp:
        data = json.load(rp)

    for i in data:
        source_text.append(i["conversations"][0]["value"])
        target_text.append(i["conversations"][1]["value"])
        source_image.append(i["image"])
            
    return source_text, source_image, target_text

_SWIPE_DISTANCE_THRESHOLD = 0.04
def is_tap_action(normalized_start_yx, normalized_end_yx):
    distance = jnp.linalg.norm(
        jnp.array(normalized_start_yx) - jnp.array(normalized_end_yx))
    return distance <= _SWIPE_DISTANCE_THRESHOLD

def _check_drag_actions_match(
    drag_touch_yx,
    drag_lift_yx,
):
    """Determines if two drag actions are the same."""
    # Store drag deltas (the change in the y and x coordinates from touch to
    # lift), magnitudes, and the index of the main axis, which is the axis with
    # the greatest change in coordinate value (e.g. a drag starting at (0, 0) and
    # ending at (0.3, 0.5) has a main axis index of 1).
    drag_1_deltas = drag_lift_yx - drag_touch_yx
    drag_1_magnitudes = jnp.abs(drag_1_deltas)
    drag_1_main_axis = np.argmax(drag_1_magnitudes)

    # y axis
    if drag_1_main_axis == 0:
        if drag_1_deltas[0] < 0:
            scroll = "up"
        else:
            scroll = "down"
    elif drag_1_main_axis == 1:
        if drag_1_deltas[1] < 0:
            scroll = "left"
        else:
            scroll = "right"
            
    return scroll

# class ScienceQADatasetImg(Dataset):
#     """
#     Creating a custom dataset for reading the dataset and
#     loading it into the dataloader to pass it to the
#     neural network for finetuning the model

#     """

#     def __init__(
#         self, data, tokenizer, source_len, target_len
#     ):
#         """
#         Initializes a Dataset class

#         Args:
#             dataframe (pandas.DataFrame): Input dataframe
#             tokenizer (transformers.tokenizer): Transformers tokenizer
#             source_len (int): Max length of source text
#             target_len (int): Max length of target text
#             source_text (str): column name of source text
#             target_text (str): column name of target text
#         """
#         self.tokenizer = tokenizer
#         self.source_len = source_len
#         self.summ_len = target_len
#         self.source_text = data[0]
#         self.source_image = data[1]
#         self.target_text = data[2]
#         self.anno_positions = data[3]
            
#     def __len__(self):
#         """returns the length of dataframe"""
#         return len(self.target_text)

#     def __getitem__(self, index):
#         """return the input ids, attention masks and target ids"""

#         source_text = str(self.source_text[index])
#         source_image = self.source_image[index]
#         target_text = str(self.target_text[index])

        
#         target_dict = eval("{" + target_text + "}")
#         action = action_type.ActionType[target_dict["action_type"]].value

#         touch_point = eval(target_dict["touch_point"])
#         lift_point = eval(target_dict["lift_point"])

#         # cleaning data so as to ensure data is in string type
#         source_text = " ".join(source_text.split())
#         target_text = " ".join(target_text.split())

#         source = self.tokenizer.batch_encode_plus(
#             [source_text],
#             max_length=self.source_len,
#             pad_to_max_length=True,
#             truncation=True,
#             padding="max_length",
#             return_tensors="pt",
#         )
#         target = self.tokenizer.batch_encode_plus(
#             [target_text],
#             max_length=self.summ_len,
#             pad_to_max_length=True,
#             truncation=True,
#             padding="max_length",
#             return_tensors="pt",
#         )

#         source_ids = source["input_ids"].squeeze()
#         source_mask = source["attention_mask"].squeeze()
#         target_ids = target["input_ids"].squeeze()
        
#         image_ids = torch.tensor(source_image).squeeze()
#         vis_attention_mask = torch.tensor([1]).squeeze()

#         act_ids = torch.tensor(action).squeeze()
#         touch_point = torch.tensor(touch_point).squeeze()
#         lift_point = torch.tensor(lift_point).squeeze()
        
        
#         return {
#             "input_ids": source_ids,
#             "attention_mask": source_mask,
#             "image_ids": image_ids,
#             "labels": target_ids,
#             "target_act": act_ids,
#             "target_touch": touch_point,
#             "target_lift": lift_point
#         }

def load_for_owl(inputfile, split, foreval=False, margs = None):
    # args = parse_args()
    class theargs: 
        debug_num = None
        data_ratio = None
        use_history = False
        use_img_history = False
        img_dim = 512
        use_layout = False
        data_root = 'general_parsed_episode_owl'
        data_path = '/data/maxb/mmcot2/dataset/owl'
        eval_subset = '/data/maxb/mmcot2/dataset/owl/general_parsed_episode_owl/'
        all_data = None
        transform_axis = True
    args = theargs()
    if margs:
        args.debug_num = margs.debug_num if margs.debug_num else args.debug_num
        args.data_ratio = margs.data_ratio if  margs.data_ratio else args.data_ratio
        args.use_history = margs.use_history if margs.use_history else args.use_history
        args.use_img_history = margs.use_img_history if margs.use_img_history else args.use_img_history
        args.img_dim = margs.img_dim if margs.img_dim else args.img_dim
        args.use_layout = margs.use_layout if margs.use_layout else args.use_layout
        args.data_root = margs.data_categ if margs.data_categ else args.data_categ
        args.eval_subset = margs.eval_subset if margs.eval_subset else args.eval_subset
        args.all_data = margs.all_data if margs.all_data else args.all_data
        args.transform_axis = margs.transform_axis if margs.transform_axis else args.transform_axis
        args.data_path = margs.data_path if margs.data_path else args.data_path

    print("args",args)
    source_text, source_image, target_text = load_data(args, split)
    assert len(source_text) == len(source_image) == len(target_text)
    # source_image = [ Image.open(img).convert('RGB') for img in source_image ]
#     text_template_1 = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
# Human: Given a mobile screen and a goal, provide the action based on the screen information. Screen: <image> Goal: <goal_info> 
# AI: <label_action>'''
    text_template_1 = '''Human: Given a mobile screen, a question and a plan, at the same time you can see the Previous Steps and Previous Actions.\n With this information, you can see what action should be taken now. The available actions are given below.\nAvailable Actions: {'action_type': 'click', 'idx': <element_idx>}\n{'action_type': 'type', 'text': <text>}\n{'action_type': 'navigate_home'}\n{'action_type': 'navigate_back'}\n{'action_type': 'scroll', 'direction': 'up'}\n{'action_type': 'scroll', 'direction': 'down'}\n{'action_type': 'scroll', 'direction': 'left'}\n{'action_type': 'scroll', 'direction': 'right'}\n{'action_type': 'status_complete'}\n{'action_type': 'press_enter'}.\n <input>
AI: <label_action>'''
    data = []
        
    for i in range(len(source_text)):
        if args.debug_num and i > args.debug_num:
            break
        di = {
            'image': source_image[i],

            'target_text': target_text[i],
            # 'anno_pos': anno_positions[i],
            "task_type": "llava_sft",
            
        }
        # source_texti = text_template_1.replace('<goal_info>', goals[i])
        # source_texti = source_texti.replace('<label_action>', target_text[i])
        source_texti = text_template_1.replace('<input>', source_text[i])
        di['text'] = source_texti
        data.append(di)
        # print(di)
    return data
    # with jsonlines.open('/data/maxb/mmcot2/dataset/owl/general_parsed_episode_owl_train.jsonl', 'w') as f:
    #     for line in data:
    #         f.write(line)
    # if not foreval:
    #     return data
    # elif args.debug_num:
    #     return anno_positions[:args.debug_num]
    # else:
    #     return anno_positions