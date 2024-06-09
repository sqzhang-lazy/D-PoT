import json
dataset_name="install"
print(dataset_name)
with open(f"llava_{dataset_name}_finetunewaction_action_v4.json", 'r') as rp:
    pred_data = json.load(rp)

with open(f"data/{dataset_name}_llava.json", 'r') as rp:
    target_data = json.load(rp)

target_data = [conv["conversations"][1]["value"] for conv in target_data]
correct = 0
for pred, target in zip(pred_data, target_data):
    try:
        pred = eval(pred)
        target = eval(target)
    except Exception as e:
        print(e)
        continue
    if target["action_type"] == pred["action_type"]:
        if pred["action_type"] == "click":
            pred_idx = pred["idx"][0]
            golden_idx = target["idx"]
            check_match = False
            if pred_idx in golden_idx:
                check_match = True

            if check_match:
                correct += 1
                # click_correct += 1      
        elif pred["action_type"] == "scroll":
            if pred["direction"] == target["direction"]:
                # scroll_correct += 1
                correct += 1
        elif pred["action_type"] == "type":
            # type_correct += 1
            correct += 1
        elif pred["action_type"] == "status_complete":
            correct += 1
            # complete_correct += 1

        elif pred["action_type"] == "navigate_home":
            # navigate_correct += 1
            correct += 1
        else:
            correct += 1

print(f"{'{:.2f}'.format(correct/len(pred_data)*100)}")
