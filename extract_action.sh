# dataset_name=general

# python extract_action.py --predict_dir result/action/${dataset_name}_SeAAgent_action.json --parsed_dir parsed_json/${dataset_name}_parsed_episode_sample.json --output_dir result/action/${dataset_name}_SeAAgent_action_ex.json

# python extract_action.py --predict_dir result/baseline/${dataset_name}_baseline_response.json --parsed_dir parsed_json/${dataset_name}_parsed_episode_sample.json --output_dir result/baseline/${dataset_name}_baseline_ex.json


# dataset_name=single

# python extract_action.py --predict_dir result/action/${dataset_name}_SeAAgent_action.json --parsed_dir parsed_json/${dataset_name}_parsed_episode_sample.json --output_dir result/action/${dataset_name}_SeAAgent_action_ex.json

# python extract_action.py --predict_dir result/baseline/${dataset_name}_baseline_response.json --parsed_dir parsed_json/${dataset_name}_parsed_episode_sample.json --output_dir result/baseline/${dataset_name}_baseline_ex.json


dataset_name=install

python extract_action.py --predict_dir result/action/${dataset_name}_SeAAgent_action.json --parsed_dir parsed_json/${dataset_name}_parsed_episode_sample.json --output_dir result/action/${dataset_name}_SeAAgent_action_ex.json

# python extract_action.py --predict_dir result/baseline/${dataset_name}_baseline_response.json --parsed_dir parsed_json/${dataset_name}_parsed_episode_sample.json --output_dir result/baseline/${dataset_name}_baseline_ex.json


# dataset_name=web_shopping

# python extract_action.py --predict_dir result/action/${dataset_name}_SeAAgent_action_human.json --parsed_dir parsed_json/${dataset_name}_parsed_episode_sample.json --output_dir result/action/${dataset_name}_SeAAgent_action_human_ex.json

# python extract_action.py --predict_dir result/baseline/${dataset_name}_baseline_response.json --parsed_dir parsed_json/${dataset_name}_parsed_episode_sample.json --output_dir result/baseline/${dataset_name}_baseline_ex.json

# dataset_name=google_apps

# python extract_action.py --predict_dir result/action/${dataset_name}_SeAAgent_action.json --parsed_dir parsed_json/${dataset_name}_parsed_episode_sample.json --output_dir result/action/${dataset_name}_SeAAgent_action_ex.json

# python extract_action.py --predict_dir result/baseline/${dataset_name}_baseline_response.json --parsed_dir parsed_json/${dataset_name}_parsed_episode_sample.json --output_dir result/baseline/${dataset_name}_baseline_ex.json




#test human plan

# dataset_name=web_shopping

# python extract_action.py --predict_dir result/action/${dataset_name}_SeAAgent_action_human.json --parsed_dir parsed_json/${dataset_name}_parsed_episode_sample_human.json --output_dir result/action/${dataset_name}_SeAAgent_action_human_ex.json

# python extract_action.py --predict_dir test/${dataset_name}_SeAAgent_action_20.json --parsed_dir parsed_json/${dataset_name}_parsed_episode_sample_human.json --output_dir test/${dataset_name}_SeAAgent_action_ex20.json