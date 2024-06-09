# dataset_name=general

# CUDA_DEVICES_VISIBLE=0 python eval.py --predict_dir result/action/${dataset_name}_SeAAgent_action_ex.json --golden_dir sample_test/${dataset_name}_test_target.json --threshold 0.14 --output_dir result/metrics/${dataset_name}_metrics.json


# CUDA_DEVICES_VISIBLE=0 python eval.py --predict_dir result/baseline/${dataset_name}_baseline_ex.json --golden_dir sample_test/${dataset_name}_test_target.json --threshold 0.14 --output_dir result/metrics/${dataset_name}_metrics_baseline.json

# dataset_name=google_apps

# CUDA_DEVICES_VISIBLE=0 python eval.py --predict_dir result/action/${dataset_name}_SeAAgent_action_ex.json --golden_dir sample_test/${dataset_name}_test_target.json --threshold 0.14 --output_dir result/metrics/${dataset_name}_metrics.json


# CUDA_DEVICES_VISIBLE=0 python eval.py --predict_dir result/baseline/${dataset_name}_baseline_ex.json --golden_dir sample_test/${dataset_name}_test_target.json --threshold 0.14 --output_dir result/metrics/${dataset_name}_metrics_baseline.json

dataset_name=install

CUDA_DEVICES_VISIBLE=0 python eval.py --predict_dir result/action/${dataset_name}_SeAAgent_action_ex.json --golden_dir sample_test/${dataset_name}_test_target.json --threshold 0.14 --output_dir result/metrics/${dataset_name}_metrics.json


# CUDA_DEVICES_VISIBLE=0 python eval.py --predict_dir result/baseline/${dataset_name}_baseline_ex.json --golden_dir sample_test/${dataset_name}_test_target.json --threshold 0.14 --output_dir result/metrics/${dataset_name}_metrics_baseline.json


# dataset_name=single

# CUDA_DEVICES_VISIBLE=0 python eval.py --predict_dir result/action/${dataset_name}_SeAAgent_action_ex.json --golden_dir sample_test/${dataset_name}_test_target.json --threshold 0.14 --output_dir result/metrics/${dataset_name}_metrics.json


# CUDA_DEVICES_VISIBLE=0 python eval.py --predict_dir result/baseline/${dataset_name}_baseline_ex.json --golden_dir sample_test/${dataset_name}_test_target.json --threshold 0.14 --output_dir result/metrics/${dataset_name}_metrics_baseline.json

# dataset_name=web_shopping

# CUDA_DEVICES_VISIBLE=0 python eval.py --predict_dir result/action/${dataset_name}_SeAAgent_action_human_ex.json --golden_dir sample_test/${dataset_name}_test_target.json --threshold 0.14 --output_dir result/metrics/${dataset_name}_human_metrics.json


# CUDA_DEVICES_VISIBLE=0 python eval.py --predict_dir result/baseline/${dataset_name}_baseline_ex.json --golden_dir sample_test/${dataset_name}_test_target.json --threshold 0.14 --output_dir result/metrics/${dataset_name}_metrics_baseline.json


# test human plan
# dataset_name=web_shopping

# CUDA_DEVICES_VISIBLE=0 python eval.py --predict_dir result/action/${dataset_name}_SeAAgent_action_human_ex.json --golden_dir sample_test/${dataset_name}_test_target_human.json --threshold 0.14 --output_dir result/metrics/${dataset_name}_human_metrics.json

# CUDA_DEVICES_VISIBLE=0 python eval.py --predict_dir test/${dataset_name}_SeAAgent_action_ex20.json --golden_dir sample_test/${dataset_name}_test_target_human.json --threshold 0.14 --output_dir result/metrics/${dataset_name}_ex20_metrics.json
