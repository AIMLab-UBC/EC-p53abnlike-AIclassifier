# Artificial intelligence-based histopathology image analysis identifies a novel subset of endometrial cancers with distinct genomic features and unfavourable outcome

### Development Information ###
```
Date Created: 13 March 2023
Developer: Amirali Darbandsari
Version: 0.0
```

### About The Project ###
The following GIF depicts the proposed workflow in our study. This repo implements the AI step of the workflow.

![](gif/workflow.gif)


## Installation

```
mkdir AttentionMIL
cd AttentionMIL
git clone git clone https://github.com/AIMLab-UBC/EC2023 .
pip install -r requirements.txt
```

### Usage ###

From a high-level perspective, AttentionMIL can be divided into three stages:
1. Deriving embeddings from patches
2. Training/Evaluating the network

Each subsection below includes sample configurations.

<details>
<summary>
Deriving the embeddings
</summary>

Following script, provide sample settings to calculate embeddings the extracted patches:
``` python
python3 run.py --experiment_name exp_name \
--log_dir path_to_dir \
--chunk_file_location path_to_json_file \
--patch_pattern pattern_of_patches \
--subtypes subtypes \
--num_classes nb_subtypes \
--backbone resnet34 \
calculate-representation \
--method Vanilla \
--saved_model_location path_to_trained_network \
```
The above configuration generates the embeddings of the extracted patches using the trained network whose weights located at `saved_model_location` and writes them in the directory located at `path_to_dir/exp_name/representation` as pickle files.
</details>

<details>
<summary>
Training the network
</summary>

Following script, provide sample settings to train the AttentionMIL:
``` python
python3 run.py --experiment_name exp_name \
--log_dir path_to_dir \
--chunk_file_location path_to_json_file \
--patch_pattern pattern_of_patches \
--subtypes subtypes \
--num_classes nb_subtypes \
--backbone resnet34 \
train-attention \
--lr 0.0001 \
--wd 0.00001 \
--epochs 30 \
--optimizer Adam \
--patience 10 \
--lr_patience 5 \
--use_schedular \
VarMIL
```
</details>

<details>
<summary>
Evaluating the network
</summary>

Following script, provide sample settings to test the trained network:
``` python
python3 run.py --experiment_name exp_name \
--log_dir path_to_dir \
--chunk_file_location path_to_json_file \
--patch_pattern pattern_of_patches \
--subtypes subtypes \
--num_classes nb_subtypes \
--backbone resnet34 \
train-attention \
--only_test \
VarMIL
```
The network generates a `.pkl` file at `path_to_dir/exp_name/information/VarMIL` consisting of predictions, attention mappings, and evaluation metrics such as AUC, accuracy, and F1-score.
</details>

### Use AttentionMIL on your data ###

To run AttentionMIL on your own data, you simply need to generate a `json` file containing the path of extracted patches.

</details>

<details>
<summary>
Sample JSON
</summary>

Each file consists of three IDs (`0`, `1`, and `2`), with `0` representing training data, `1` representing validation data, and `2` representing test data.
```
{"chunks": [{"id": 0, "imgs": ["pattern_of_patches/x1_y1.png", "pattern_of_patches/x2_y2.png"]}, {"id": 1, "imgs": ["pattern_of_patches/x3_y3.png", "pattern_of_patches/x4_y4.png"]}, {"id": 2, "imgs": ["pattern_of_patches/x5_y5.png", "pattern_of_patches/x6_y6.png"]}]}
```
</details>
