import json

with open("google_apps_input.json", 'r') as rp:
    data = json.load(rp)


with open("google_plan.json", 'r') as rp:
    ori_target = json.load(rp)

target = []
source = []
for i in data:
    ep_idx = i["episode_id"][:40]
    for t in ori_target:
        t_idx = t["episode_id"]
        if t_idx == ep_idx:
            target.append(t)
            source.append(i)
            break

print(len(target))

aitw_list = []

# for i, (instruction, outputs) in enumerate(zip(source, target)):
#     img_ins_dict = {}
#     t_idx = outputs["episode_id"]
#     d_idx = instruction["episode_id"][:40]
#     print(t_idx,d_idx)
#     if t_idx != d_idx:
#         print(f"wrong {t_idx}")
#         exit()
#     img_ins_dict["id"] = t_idx
#     ins_data = instruction["data"]
#     out_data = outputs["result"]
#     for i, ((_, input), (_, output)) in enumerate(zip(ins_data.items(), out_data.items())):
#         # input = input.split("And the previous step")[0]
#         output = eval(output)["parsed_action"]
#         Imgstep_url = f"new_ImageDataset/single_ScreenImg/{t_idx}/step {i}.png"

#         aitw_list.append({
#             "id": t_idx,
#             "image": Imgstep_url,
#             "conversations": [
#                 {'from': 'human', 'value': f"{input}"},
#                 {'from': 'gpt', 'value': f"{output}"},
#             ]
#         })

for i, (instruction, outputs) in enumerate(zip(source, target)):
    img_ins_dict = {}
    t_idx = outputs["episode_id"]
    d_idx = instruction["episode_id"][:40]
    # print(t_idx,d_idx)
    if t_idx != d_idx:
        print(f"wrong {t_idx}")
        exit()
    img_ins_dict["id"] = t_idx
    ins_data = instruction["data"]
    out_data = outputs["data"]
    for i, ((_, input), (_, output)) in enumerate(zip(ins_data.items(), out_data.items())):
        # input = input.split("And the previous step")[0]
        # output = eval(output)["parsed_action"]
        input = input.split("Plan:")[0] + "Please formulate an operational guide for future operations for solving the goal. The guide includes:" + \
            "1. Plan: A **multi-step future** plan **(start from current screen, DON'T include previous steps)**; steps indexed by numbers." + \
            "2. Step: Based on the current screen and Previous Steps, provide the **immediate** step that needs to be taken from the Plan.\n" + \
            "**Output Format:** A JSON dictionary strictly following the format:\n" + \
            "{'plan': '...<Your Plan Here>', 'step': '...<Your Step Here>'}\n" + \
            "If the goal has already been implemented, no more planning is required, Provide {'plan': '1. Mark the task as complete', 'step': 'Mark the task as complet'}" + \
            "**Please do not output any content other than the JSON format.**"
        Imgstep_url = f"new_ImageDataset/google_apps_ScreenImg/{t_idx}/step {i}.png"

        aitw_list.append({
            "id": t_idx,
            "image": Imgstep_url,
            "conversations": [
                {'from': 'human', 'value': f"{input}"},
                {'from': 'gpt', 'value': f"{output}"},
            ]
        })
    
with open("./data/google_apps_plan.json", "w") as wp:
    json.dump(aitw_list, wp, indent=4)


