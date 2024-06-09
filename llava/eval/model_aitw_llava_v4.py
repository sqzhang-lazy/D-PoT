import sys

sys.path.append("/cpfs/29bcf0bdae829206-000001/home/usera409/AAgent-main")

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from utils_data_for_owl import load_for_owl
from llava.train.train import * 

from PIL import Image
import math
import action_type
import action_matching



# os.environ["CUDA_VISIBLE_DEVICES"] = '4'

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def preprocess_multimodal(
    sources: Sequence[str],
    # data_args: DataArguments
) -> Dict:
    # is_multimodal = data_args.is_multimodal
    # if not is_multimodal:
    #     return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                # why put the image in front of the sentence?
                # sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                # sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if 1==1:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def eval_model(args):
    # Model
    disable_torch_init()
    from transformers import AutoProcessor, AutoModelForCausalLM, LlavaForConditionalGeneration, AutoTokenizer

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path
    )
    model.to('cuda')
    print(model)
    # model_path = os.path.expanduser(args.model_path)
    # print(model_path)
    # model_name = get_model_name_from_path(model_path)
    # print(model_name)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # data = load_for_owl('.', 'test')
    # questions = data
    with open(args.question_file, 'r') as rp:
        questions = json.load(rp)
    # questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # answers_file = os.path.expanduser(args.answers_file)
    # answers_file = "answer-file-our.json"
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "w")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=3072,
        padding_side="right",
    )
    preds = []
    actions = []
    targets = []
    metrics = {}
    partial_correct = 0
    text_correct = 0
    type_correct = 0
    reference_test_positions = []

    processor = AutoProcessor.from_pretrained("llava-7b-hf")

    for i, line in enumerate(tqdm(questions)):
        # reference_test_positions.append(data[i]['anno_pos'])
        # targets.append(data[i]['target_text'])
        # print('assert: ', targets[-1], line['conversations'][-1]['value'])
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].strip().split("Plan")[0]+"Please formulate an operational guide for future operations for solving the goal. The guide includes:" + \
            "1. Plan: A **multi-step future** plan **(start from current screen, DON'T include previous steps)**; steps indexed by numbers." + \
            "2. Step: Based on the current screen and Previous Steps, provide the **immediate** step that needs to be taken from the Plan.\n" + \
            "**Output Format:** A JSON dictionary strictly following the format:\n" + \
            "{'plan': '...<Your Plan Here>', 'step': '...<Your Step Here>'}\n" + \
            "If the goal has already been implemented, no more planning is required, Provide {'plan': '1. Mark the task as complete', 'step': 'Mark the task as complet'}" + \
            "**Please do not output any content other than the JSON format.**"
        cur_prompt = qs
        # print("original dataset: ", line['conversations'])
        gt = line['conversations'][1]['value']
        if 'image' in line:
            image_file = line["image"]
            
            if type(image_file) == list:
                image = [ Image.open(os.path.join(args.image_folder, f)) for f in image_file ] 
            else:
                image = Image.open(os.path.join(args.image_folder, image_file))
            
            # image_return = processor(
            #     images = image,
            #     return_tensors="pt"
            # )
            # #  only pick the first image
            # if type(image_file) == list:
            #     image = image_return['pixel_values']
            # else:
            #     image = image_return['pixel_values'][0]
            # images = image.unsqueeze(0).half().cuda()
            # if getattr(model.config, 'mm_use_im_start_end', False):
            if 1==1:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            qs = question['value'].strip()
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        # print(f"conv {conv}")
        prompt = conv.get_prompt()
        # print(prompt)
        # prompt = preprocess_multimodal([prompt])

        # data_dict = preprocess(
        #     prompt,
        #     tokenizer,
        #     has_image=('image' in self.list_data_dict[i]))
        prompt = '<image>\n' + prompt
        inputs_cpu = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {}
        # inputs["input_ids"] = inputs_cpu["input_ids"].cuda()
        for key, value in inputs_cpu.items():
            inputs[key] = value.cuda()
        # print(inputs)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=300,
            )

        input_token_len = inputs["input_ids"].shape[1]

        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        try:
            plan_dict = eval(outputs)
            step = plan_dict["step"]
        except Exception as e:
            print(e)
            step = ""
        qs = question['value'].replace('<image>', '').strip().split("Plan")[0]+"Plan: "+step
        cur_prompt = qs

        gt = line['conversations'][1]['value']
        if 'image' in line:
            # image_file = line["image"]
            
            # if type(image_file) == list:
            #     image = [ Image.open(os.path.join(args.image_folder, f)) for f in image_file ] 
            # else:
            #     image = Image.open(os.path.join(args.image_folder, image_file))
            
            # image_return = processor(
            #     images = image,
            #     return_tensors="pt"
            # )
            # #  only pick the first image
            # if type(image_file) == list:
            #     image = image_return['pixel_values']
            # else:
            #     image = image_return['pixel_values'][0]
            # images = image.unsqueeze(0).half().cuda()
            # if getattr(model.config, 'mm_use_im_start_end', False):
            if 1==1:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            qs = question['value'].strip()
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        # print(f"conv {conv}")
        prompt = conv.get_prompt()

        prompt = '<image>\n' + prompt
        inputs_cpu = processor(text=prompt, images=image, return_tensors="pt")
        
        inputs = {}


        for key, value in inputs_cpu.items():
            inputs[key] = value.cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=300,
            )

        input_token_len = inputs["input_ids"].shape[1]

        action = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        action = action.strip()
        if action.endswith(stop_str):
            action = action[:-len(stop_str)]
        action = action.strip()

        preds.append(outputs)
        actions.append(action)
        
        with open(args.answers_file, 'w') as w:
            json.dump(preds, w, indent=4)
        
        with open(args.answers_file.replace("plan", "action"), 'w') as w:
            json.dump(actions, w, indent=4)
    # ans_file.close()
    
    print('file closed')
    # # metrics
    # output_data = []

    # assert len(preds) == len(targets)   == len(reference_test_positions)
    with open(args.answers_file, 'w') as w:
        json.dump(preds, w, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="liuhaotian/llava-lcs558k-scienceqa-vicuna-13b-v1.3")
    parser.add_argument("--model_path", type=str, default="/data/maxb/tag/LLaVA/checkpoints/llava-CONtinue_ep1_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-12000")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--question_file", type=str, default="/data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json")
    parser.add_argument("--answers_file", type=str, default="./res_continue/12k_ep1_llava_try-llama-2-7b-chat-finetune.jsonl")
    parser.add_argument("--conv_mode", type=str, default="llava_llama_2")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--answer_prompter", action="store_true")
    # parser.add_argument("--prd_output_path", type=str, default='.')
    # parser.add_argument("--eval_name", type=str, default=None)
    # parser.add_argument("--eval_data", type=str, default='/data/maxb/mmcot2/dataset/owl/general_parsed_episode_owl_test.obj')
    # parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()
    print(args)
    
    eval_model(args)

