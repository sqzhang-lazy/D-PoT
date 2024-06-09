from openai import OpenAI

import json
import base64
import re
import os
from tqdm import tqdm
import ast


os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  

import requests


if not os.path.exists(f"test"):
    os.mkdir(f"test")

if not os.path.exists(f"test/action"):
    os.mkdir(f"test/action")
if not os.path.exists(f"test/plan"):
    os.mkdir(f"test/plan")
if not os.path.exists(f"test/step_input"):
    os.mkdir(f"test/step_input")

#GPT 4
dataset_name = "web_shopping"


with open(f"new_sample_test/test/{dataset_name}_test_source.json", "r", encoding="utf-8") as rp:
   _data = json.load(rp)


with open("prompt/prompts2.json", 'r') as rp:
    prompts = json.load(rp)


sys_prompt_make_plan_and_select_step = prompts["sys_prompt_make_plan_and_select_step"]


sys_prompt_action = prompts["sys_prompt_action"]


    # "If you have more than one action option, please give them all.\n" + \ 

messages_action = []
messages_action.append(
    {
        "role": "system",
        "content": [
            {"type": "text", "text": sys_prompt_action}
        ]
    }
)

messages_make_plan_and_select_step = []
messages_make_plan_and_select_step.append(
    {
        "role": "system",
        "content": [
            {"type": "text", "text": sys_prompt_make_plan_and_select_step}
        ]
    }
)


client = OpenAI(
    api_key="",
)


# result solve list
new_eps_planAsteps_list = []
new_eps_planAsteps_list.append({
    "sys_prompt_make_plan prompt": sys_prompt_make_plan_and_select_step,
    "sys_prompt_action": sys_prompt_action,
})

new_eps_step_input_list = []
new_eps_action_list = []
_data = _data[:]

token_number = 0

try:
    for i, ep in tqdm(enumerate(_data)):
        new_ep_planAsteps_dict = {}
        new_ep_step_input_dict = {}
        new_ep_action_dict = {}
        episode_id = ep.get("episode_id")
        print(f"episode_id: {episode_id}")

        ep_steps = ep.get("data")
        _previous_steps = "Previous Steps:\n"
        _previous_actions = "Previous Actions:\n"

        planAsteps_dict = {}
        step_input_dict = {}
        action_dict = {}

        for step_count, step_input in ep_steps.items():
            print(f"This is step {step_count}")
            # 拼接step
            new_step_input = f"{_previous_steps}{step_input}"

            # 获取Plan和Step
            Imgstep_url = f"new_ImageDataset/{dataset_name}_ScreenImg/{episode_id}/step {step_count}.png"
            base64_image = encode_image(Imgstep_url)

            step_message = messages_make_plan_and_select_step.copy()

            # match_obj = re.search(r"Previous(?:.|\n)*(?=Goal)", step_input)
            # _previous_actions = match_obj.group()
            match_obj = re.search(r"Screen(?:.|\n)*", step_input)
            _screen = match_obj.group()
            match_obj = re.search(r"(?<=Goal: ).*(?=\nScreen)", step_input)
            _goal = match_obj.group()
            
            text_prompt = f"**Your ultimate goal is: {_goal}.**\nThe current on-screen input is:\n{_screen}\n" + \
            f"Here are previous actions: (format: action → action description)\n{_previous_actions}\nAnd the previous steps:\n{_previous_steps}\n" + \
            "Please formulate an operational guide for future operations for solving the goal. The guide includes:" + \
            "1. Plan: A **multi-step future** plan **(start from current screen, DON'T include previous steps)**; steps indexed by numbers." + \
            "2. Step: Based on the current screen and Previous Steps, provide the **immediate** step that needs to be taken from the Plan.\n" + \
            "**Output Format:** A JSON dictionary strictly following the format:\n" + \
            "{'plan': '...<Your Plan Here>', 'step': '...<Your Step Here>'}\n" + \
            "If the goal has already been implemented, no more planning is required, Provide {'plan': '1. Mark the task as complete', 'step': 'Mark the task as complet'}" + \
            "**Please do not output any content other than the JSON format.**"

            step_message.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {
                    "type": "image_url",
                    "image_url":{
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    }
                }
            ]
            })

            chat_completion = client.chat.completions.create(
                messages=step_message,
                model="gpt-4-vision-preview",
                max_tokens=300,
                temperature=0,
            )
            res = chat_completion.choices[0].message.content
            token_number += chat_completion.usage.total_tokens


            try:
                match_obj = re.search(r"{(?:.|\n)*}", res)
                res_dict = match_obj.group()
                match_obj = re.search(r"(?<=('|\")plan('|\"): ('|\")).*(?=('|\"),)", res_dict)
                plan = match_obj.group()
                match_obj = re.search(r"(?<=('|\")step('|\"): ('|\")).*(?=('|\"))", res_dict)
                step = match_obj.group()
                # res_dict = ast.literal_eval(res_dict)
            except:
                print("response wrong")
                print(res)
                step = ""
                plan = ""
            

            print(f"Get the action of step {step_count}")
            text_prompt = f"**Your ultimate goal is: {_goal}.**\nThe current on-screen input is:\n{_screen}\n" + \
            f"Here are previous actions: (format: action → action description)\n{_previous_actions}\nAnd the previous steps:\n{_previous_steps}\n"

            step_message = messages_action.copy()
            text_prompt = f"{text_prompt}Plan:{step}"

            step_message.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {
                    "type": "image_url",
                    "image_url":{
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    }
                }
            ]
            })
            # print(_previous_steps)
            chat_completion = client.chat.completions.create(
                messages=step_message,
                model="gpt-4-vision-preview",
                max_tokens=300,
                temperature=0,
            )
            response = chat_completion.choices[0].message.content
            token_number += chat_completion.usage.total_tokens


            _previous_steps += f"Step {int(step_count)+1}. {step}\n"
            _previous_steps = _previous_steps.replace("\n\n", "\n")

            pattern = re.compile(r"\{.*\}")
            result = pattern.findall(response)
            if len(result):
                extract_action = result[-1]
                extract_action = extract_action.replace('"', "'")
                try:
                    _action_dict = eval(extract_action)
                except Exception as e:
                    print(e)
                    print(extract_action)
                    _action_dict = {"action_type": "no action"}
            
            _previous_actions += str(_action_dict)

            # _previous_actions = 

            planAsteps_dict[step_count] = res_dict
            step_input_dict[step_count] = text_prompt
            action_dict[step_count] = response
        
        new_ep_planAsteps_dict["episode_id"] = episode_id
        new_ep_planAsteps_dict["data"] = planAsteps_dict
        new_eps_planAsteps_list.append(new_ep_planAsteps_dict)

        new_ep_step_input_dict["episode_id"] = episode_id
        new_ep_step_input_dict["data"] = step_input_dict
        new_eps_step_input_list.append(new_ep_step_input_dict)

        new_ep_action_dict["episode_id"] = episode_id
        new_ep_action_dict["response"] = action_dict
        new_eps_action_list.append(new_ep_action_dict)
        

        with open(f"test/action/{dataset_name}_DPAgent_action.json", 'w') as f:
            json.dump(new_eps_action_list, f, indent=4)



except Exception as e:
    print(e)
    new_ep_planAsteps_dict["episode_id"] = episode_id
    new_ep_planAsteps_dict["data"] = planAsteps_dict
    new_eps_planAsteps_list.append(new_ep_planAsteps_dict)

    new_ep_step_input_dict["episode_id"] = episode_id
    new_ep_step_input_dict["data"] = step_input_dict
    new_eps_step_input_list.append(new_ep_step_input_dict)

    new_ep_action_dict["episode_id"] = episode_id
    new_ep_action_dict["response"] = action_dict
    new_eps_action_list.append(new_ep_action_dict)



with open(f"test/action/{dataset_name}_DPAgent_action.json", 'w') as f:
    json.dump(new_eps_action_list, f, indent=4)

with open(f"test/plan/{dataset_name}_DPAgent_plan.json", 'a') as f:
    json.dump(new_eps_planAsteps_list, f, indent=4)

with open(f"test/step_input/{dataset_name}_DPAgent_new_step_input.json", 'a') as f:
    json.dump(new_eps_step_input_list, f, indent=4)

print(token_number)




