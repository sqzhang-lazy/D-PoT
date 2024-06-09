import argparse
import json
import action_matching, action_type
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_dir', type=str, default='extract_baseline.json')
    parser.add_argument("--output_dir", type=str, default="result/metrics/metrics.json")
    parser.add_argument('--golden_dir', type=str, default='sample_test/general_test_target.json')
    parser.add_argument("--threshold", type=float, default='0.16')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    with open(args.predict_dir, 'r') as f:
        predict_ans = json.load(f)

    with open(args.golden_dir, 'r') as f:
        golden_ans = json.load(f)

    threshold = args.threshold**2

    metrics = {}
    action_correct = 0
    click_correct = 0
    scroll_correct = 0
    type_correct = 0
    complete_correct = 0
    navigate_correct = 0

    clicks_num = 0
    types_num = 0
    scrolls_num = 0
    navigate_num = 0
    completes_num = 0

    actions_num = 0

    steps_correct = {}
    
    for pred, golden in zip(predict_ans, golden_ans):
        golden_id = golden["episode_id"].replace(' ', '_').replace('&', '_')
        pred_id = pred["episode_id"].replace(' ', '_').replace('&', '_')
        if pred_id != golden_id:
            print(f"ep {pred['episode_id']} is wrong!")
            continue

        print(f"ep {pred['episode_id']}")
        pred_data = pred["data"]
        golden_data = golden["result"]
        step_correct = 0
        for (i, pred_action), (j, golden_action) in zip(pred_data.items(), golden_data.items()):
            if i != j:
                print(f"ep {pred['episode_id']}'s number of step is wrong!")
                break
            actions_num += 1
            golden_action = eval(golden_action)
            golden_parsed_action = golden_action["parsed_action"]
            # 先判断Action type
            if pred_action["action_type"] == "click":
                clicks_num += 1
                    
            elif pred_action["action_type"] == "scroll":
                scrolls_num += 1
            elif pred_action["action_type"] == "type":
                types_num += 1
            elif pred_action["action_type"] == "status_complete":
                completes_num += 1
            elif pred_action["action_type"] == "navigate_home":
                navigate_num += 1
            else:
                pass
            if golden_parsed_action["action_type"] == pred_action["action_type"]:
                if pred_action["action_type"] == "click":
                    pred_idx = int(pred_action["idx"])
                    golden_idx = golden_parsed_action["idx"]
                    pred_yx1 = pred_action["result_touch_yx1"]
                    pred_yx2 = pred_action["result_touch_yx2"]
                    pred_yx3 = pred_action["result_touch_yx3"]
                    pred_yx4 = pred_action["result_touch_yx4"]
                    pred_yx5 = pred_action["result_touch_yx5"]
                    golden_yx = golden_action["result_touch_yx"]

                    annotation_positions = np.array(pred_action["annotation_positions"])
                    check_match = False
                    if pred_idx in golden_idx:
                        check_match = True
                    else:
                        try:
                            check_match1 = action_matching.check_actions_match(
                                pred_yx1,
                                pred_yx1,
                                action_type.ActionType["DUAL_POINT"],
                                golden_yx,
                                golden_yx,
                                action_type.ActionType["DUAL_POINT"],
                                annotation_positions
                            )
                            check_match2 = action_matching.check_actions_match(
                                pred_yx2,
                                pred_yx2,
                                action_type.ActionType["DUAL_POINT"],
                                golden_yx,
                                golden_yx,
                                action_type.ActionType["DUAL_POINT"],
                                annotation_positions
                            )
                            check_match3 = action_matching.check_actions_match(
                                pred_yx3,
                                pred_yx3,
                                action_type.ActionType["DUAL_POINT"],
                                golden_yx,
                                golden_yx,
                                action_type.ActionType["DUAL_POINT"],
                                annotation_positions
                            )
                            check_match4 = action_matching.check_actions_match(
                                pred_yx4,
                                pred_yx4,
                                action_type.ActionType["DUAL_POINT"],
                                golden_yx,
                                golden_yx,
                                action_type.ActionType["DUAL_POINT"],
                                annotation_positions
                            )
                            check_match5 = action_matching.check_actions_match(
                                pred_yx5,
                                pred_yx5,
                                action_type.ActionType["DUAL_POINT"],
                                golden_yx,
                                golden_yx,
                                action_type.ActionType["DUAL_POINT"],
                                annotation_positions
                            )
                            if check_match1 or check_match2 or check_match3 or check_match4 or check_match5:
                                check_match = True

                        except Exception as exc:
                            print(exc)
                            print(i)
                            check_match = False
                            match_label = "invalid"

                    if check_match:
                        action_correct += 1
                        click_correct += 1
                        step_correct += 1
                        
                elif pred_action["action_type"] == "scroll":
                    if pred_action["direction"] == golden_parsed_action["direction"]:
                        scroll_correct += 1
                        action_correct += 1
                        step_correct += 1
                elif pred_action["action_type"] == "type":
                    type_correct += 1
                    action_correct += 1
                    step_correct += 1
                elif pred_action["action_type"] == "status_complete":
                    action_correct += 1
                    complete_correct += 1
                    step_correct += 1
                elif pred_action["action_type"] == "navigate_home":
                    navigate_correct += 1
                    action_correct += 1
                    step_correct += 1
                else:
                    action_correct += 1
                    step_correct += 1
        steps_correct[pred['episode_id']] = step_correct
        step_correct = 0
    
    metrics["accuracy"] = "{:.2f}".format(action_correct/actions_num * 100)
    metrics["click_acc"] = "{:.2f}".format(click_correct/actions_num * 100)
    metrics["scroll_acc"] = "{:.2f}".format(scroll_correct/actions_num * 100)
    metrics["type_acc"] = "{:.2f}".format(type_correct/actions_num * 100)
    metrics["navigate_acc"] = "{:.2f}".format(navigate_correct/actions_num * 100)
    metrics["complete_acc"] = "{:.2f}".format(complete_correct/actions_num * 100)

    metrics["action_correct"] = action_correct
    metrics["click_correct"] = click_correct
    metrics["scroll_correct"] = scroll_correct
    metrics["type_correct"] = type_correct
    metrics["navigate_correct"] = navigate_correct
    metrics["complete_correct"] = complete_correct


    metrics["pred_clicks_per"] = "{:.2f}".format(clicks_num/actions_num * 100)
    metrics["pred_scrolls_per"] = "{:.2f}".format(scrolls_num/actions_num * 100)
    metrics["pred_types_per"] = "{:.2f}".format(types_num/actions_num * 100)
    metrics["pred_navigate_per"] = "{:.2f}".format(navigate_num/actions_num * 100)
    metrics["pred_completes_per"] = "{:.2f}".format(completes_num/actions_num * 100)

    metrics["total_num"] = actions_num
    metrics["steps_correct_num"] = steps_correct

    with open(args.output_dir, 'w') as w:
        json.dump(metrics, w, indent=4)
                

                    
                