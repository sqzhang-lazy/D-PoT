# D-PoT
A code for Dynamic Planning for LLM-based Graphical User Interface Automation

## Test Set Sampling
According to the train/test split in the [AITW](https://github.com/google-research/google-research/tree/master/android_in_the_wild) dataset, sample the required number of trajectory data from the test set.

## Data Processing
Use the following files from the [Auto-GUI](https://github.com/cooelf/Auto-GUI) project to process Screen parsing results:
 - fetch_feature.py - Extracts screen features
 - utils_data.py - Data processing utilities

## Generating Evaluation Input Data
Refer to the load_data function in the utils_data.py file in this repository to process the golden file:

## Predict
```bash
python prompt_D-PoT_gpt4v.py
```


## Evaluation Process
1. Action Extraction
Use extract_action.py to process GPT-generated files and extract corresponding actions:

```bash
bash extract_action.sh
```
2. Result Evaluation
Use eval.py to compare predicted results with ground truth:
```bash
bash eval.sh
```
