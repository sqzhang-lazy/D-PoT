import json
import re
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_dir', type=str, default='extract_baseline.json')
    parser.add_argument("--parsed_dir", type=str, default="result/metrics/metrics.json")
    parser.add_argument('--output_dir', type=str, default='sample_test/general_test_target.json')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)

    with open(args.predict_dir, "r", encoding="utf-8") as rp:
        _data = json.load(rp)

    with open(args.parsed_dir, "r", encoding="utf-8") as rp:
        parsed_data = json.load(rp)

    # click_ans = []
    actions_list = []

    for ep, res in zip(parsed_data, _data):
        episode_id = res["episode_id"]
        # episode_id = res["episode_id"] if res["episode_id"] == ep["episode_id"] else None
        
        print(f"This is episode {episode_id}")
        
        responses = res["response"]
        datas = ep["data"]

        ep_dict = {}

        ex_action_dict = {}


        for (i, step_response), step_data in zip(responses.items(), datas):
            # SeAAgent专用
            if isinstance(step_response, dict):
                step_response =  step_response["response"]

            ui_positions = step_data["ui_positions"]

            pattern = re.compile(r"\{.*\}")
            result = pattern.findall(step_response)
            if len(result):
                extract_action = result[-1]
                extract_action = extract_action.replace('"', "'")
                try:
                    action_dict = eval(extract_action)
                except Exception as e:
                    print(e)
                    print(extract_action)
                    new_action = {"action_type": "No match action"}
                    ex_action_dict[i] = new_action
                    continue
                new_action = {}
                new_action["action_type"] = action_dict["action_type"]
                if action_dict["action_type"] == "click":
                    click_id = action_dict["idx"]
                    new_action["idx"] = int(click_id)
                    if int(click_id) < len(ui_positions):
                        top, left, height, width = ui_positions[int(click_id)]
                        bottom, right = top + height, left + width
                        x = top+height/2
                        y = left+width/2
                        new_action['result_touch_yx1'] = [top,left]
                        new_action['result_touch_yx2'] = [top,right]
                        new_action['result_touch_yx3'] = [bottom,left]
                        new_action['result_touch_yx4'] = [bottom,right]
                        new_action['result_touch_yx5'] = [x,y]
                        new_action["annotation_positions"] = ui_positions
                    else:
                        new_action["action_type"] = "No match"
                    

                elif action_dict["action_type"] == "scroll":
                    new_action["direction"] = action_dict["direction"]
                else:
                    new_action = action_dict
            else:
                new_action = {"action_type": "No match action"}
            ex_action_dict[i] = new_action

        ep_dict["episode_id"] = episode_id
        ep_dict["data"] = ex_action_dict

        actions_list.append(ep_dict)
        
        

    with open(args.output_dir, 'w') as  f:
        json.dump(actions_list, f, indent=4)

