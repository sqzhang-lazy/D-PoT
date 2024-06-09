from openai import OpenAI
from tqdm import tqdm

import json
import base64
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

if not os.path.exists(f"result60_test3"):
    os.mkdir(f"result60_test3")

if not os.path.exists(f"result60_test3/baseline"):
    os.mkdir(f"result60_test3/baseline")

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

#GPT 4
dataset_name = "google_apps"
# load test3 dataset
with open(f"new_sample_test/test3/{dataset_name}_test3_source.json", "r", encoding="utf-8") as rp:
   _data = json.load(rp)

# with open(f"new_sample_test/analyse/{dataset_name}_analyse_test_source.json", "r", encoding="utf-8") as rp:
#    _data = json.load(rp)


sys_prompt = "Given a mobile screen and a question, provide the action based on the screen information.\nAvailable Actions: {\"action_type\": \"click\", \"idx\": <element_idx>}\n" + \
    "{\"action_type\": \"type\", \"text\": <text>}\n{\"action_type\": \"navigate_home\"}\n{\"action_type\": \"navigate_back\"}\n{\"action_type\": \"scroll\", \"direction\": \"up\"}" + \
    "{\"action_type\": \"scroll\", \"direction\": \"down\"}\n{\"action_type\": \"scroll\", \"direction\": \"left\"}\n{\"action_type\": \"scroll\", \"direction\": \"right\"}" + \
    "{'action_type': 'status_complete'}\n{'action_type': 'press_enter'}"
messages = []
messages.append(
    {
        "role": "system",
        "content": [
            {"type": "text", "text": sys_prompt}
        ]
    }
)

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="",
)

result = []
_data = _data[:]
try:
    for i, ep in tqdm(enumerate(_data)):
        episode_id = ep.get("episode_id")
        ep_dict = {}
        ep_dict["episode_id"] = episode_id
        ep_data = {}
        print(f"episode id: {episode_id}")
        ep_steps = ep.get("data")
        for step_count, step_input in ep_steps.items():
            step_input = step_input
            print(f"carry out {step_count}")
            step_message = messages.copy()
            Imgstep_url = f"new_ImageDataset/{dataset_name}_ScreenImg/{episode_id}/step {step_count}.png"
            base64_image = encode_image(Imgstep_url)
            step_message.append({
            "role": "user",
            "content": [
                {"type": "text", "text": step_input},
                {
                    "type": "image_url",
                    "image_url":{
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    }
                }
            ]
            })
            # print(step_message)
            chat_completion = client.chat.completions.create(
                messages=step_message,
                model="gpt-4-vision-preview",
                max_tokens=300,
                temperature=0,
            )
            ep_data[step_count] = chat_completion.choices[0].message.content
            # ep_data[step_count] = step_input

        ep_dict["response"] = ep_data
        result.append(ep_dict)
        with open(f"result60_test3/baseline/{dataset_name}_baseline_response.json", 'w') as  f:
            json.dump(result, f, indent=4)


except Exception as e:
    print(e)
    ep_dict["response"] = ep_data
    result.append(ep_dict)

with open(f"result60_test3/baseline/{dataset_name}_baseline_response.json", 'w') as  f:
    json.dump(result, f, indent=4)


    

