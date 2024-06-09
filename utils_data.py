from torch.utils.data import Dataset
import torch
import pickle
from tqdm import tqdm
import action_matching, action_type
import numpy as np
import jax.numpy as jnp
import random
import re
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


def load_data(data):
    target_text = []
    source_text = []
    source_image = []
    anno_positions = []

    ep_list = []

    total_results_list = []

    for qid, episode in enumerate(tqdm(data)):
        ep_dict = {}
        ep_result_dict = {}
        episode_id = episode["episode_id"]
        episode_data = episode["data"]
        if True:
            history_action = []
        steps_dict = {}
        steps_result_dict = {}
        for step_idx, step_data in enumerate(episode_data):
            question = step_data["goal"]
            question = f"Goal: {question}"

            ui_positions = step_data["ui_positions"]
            ui_text = step_data["ui_text"]
            ui_type = step_data["ui_type"]

            if True:
                icon_string = ""
                for ui_idx, ui_type_i in enumerate(ui_type):

                    if ui_type_i == "TEXT":
                        icon_string += f'<p id={ui_idx} class="text" alt="{ui_text[ui_idx]}">{ui_text[ui_idx]}</p>\n'
                    elif "ICON" in ui_type_i:
                        icon_string += f'<img id={ui_idx} class={ui_type_i} alt="{ui_text[ui_idx]}">{ui_text[ui_idx]}</p>\n'
                    else:
                        print(icon_string)
                        assert "parsing ui failed!!!"
                
                question = f"{question}\nScreen: {icon_string}"
                # print(question)
            target_action = {
                "parsed_action": step_data["result_parsed_action"],
                "result_touch_yx": step_data["result_touch_yx"]
            }
            str_target = ""
            if target_action["parsed_action"]["action_type"] == "click":
                click_item = ui_text[target_action["parsed_action"]["idx"][0]]
                str_target += "{" + f'"step_idx": {step_idx}, "action_description": "{target_action["parsed_action"]["action_type"]} [{click_item}]"' + "}\n"
            elif target_action["parsed_action"]["action_type"] == "scroll":
                str_target += "{" + f'"step_idx": {step_idx}, "action_description": "{target_action["parsed_action"]["action_type"]} {target_action["parsed_action"]["direction"]}"' + "}\n"
            else:
                str_target += "{" + f'"step_idx": {step_idx}, "action_description": "{target_action["parsed_action"]["action_type"]}"' + "}\n"


            
            if True:
                prev_actions = "\n".join(history_action)
                question = f"Previous Actions: {prev_actions}\n{question}"


            if True:
                history_action.append(str_target)
            steps_dict[step_idx] = question

            steps_result_dict[step_idx] = str(target_action)


                        
        ep_dict["episode_id"] = episode_id
        ep_dict["data"] = steps_dict
        ep_list.append(ep_dict)

        ep_result_dict["episode_id"] = episode_id
        ep_result_dict["result"] = steps_result_dict
        total_results_list.append(ep_result_dict)

        if False:
            if int(qid) > args.debug_num:
                break
            
    return ep_list, total_results_list, anno_positions

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

class AITWDatasetImg(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, data, tokenizer, source_len, target_len
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.summ_len = target_len
        self.source_text = data[0]
        self.source_image = data[1]
        self.target_text = data[2]
        self.anno_positions = data[3]
            
    def __len__(self):
        """returns the length of dataframe"""
        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        source_image = self.source_image[index]
        target_text_org = str(self.target_text[index])


        # abc = self.tokenizer.tokenize(target_text)
        # print(len(abc))

        pattern = r'(?<=Action Decision:\s).*'
        result = re.search(pattern, target_text_org)
        target_text = result.group(0)
        target_text = target_text.strip()
        
        target_dict = eval("{" + target_text + "}")
        action = action_type.ActionType[target_dict["action_type"]].value

        touch_point = eval(target_dict["touch_point"])
        lift_point = eval(target_dict["lift_point"])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text_org = " ".join(target_text_org.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text_org],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        
        image_ids = torch.tensor(source_image).squeeze()
        vis_attention_mask = torch.tensor([1]).squeeze()

        act_ids = torch.tensor(action).squeeze()
        touch_point = torch.tensor(touch_point).squeeze()
        lift_point = torch.tensor(lift_point).squeeze()
        
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_ids,
            "labels": target_ids,
            "target_act": act_ids,
            "target_touch": touch_point,
            "target_lift": lift_point
        }
