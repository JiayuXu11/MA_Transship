import os
import json

# # # ----------单个base配置文件 + 单个change配置文件----------
# default_config_path = "tuned_configs/multi_lt_transship_mechanism/fine_tuned_final.json"
# cmd_format = "python train.py --load_config {} --load_change_config {} --exp_name {}"

# dir = "tuned_configs/multi_lt_transship_mechanism/offical_test/scalable"

# # 深度优先遍历文件夹下的所有json文件
# cmd_list = []
# for root, _, files in os.walk(dir):
#     for file in files:
#         if file.endswith(".json"):
#             change_config_path = os.path.join(root, file).replace("\\", "/")
#             # 读取其中main_args.exp_name的值
#             exp_name = json.load(open(change_config_path, "r", encoding='utf-8'))["main_args"]["exp_name"]
#             cmd_list.append(cmd_format.format(default_config_path.replace("\\", "/"), change_config_path, exp_name))

# # 存储为shell脚本
# with open("caogao_run_train.sh", "w") as f:
#     for cmd in cmd_list:
#         f.write(cmd + "\n")

# ----------单个base配置文件 + 多个change配置文件----------
# 单个base配置文件 + 多个change配置文件
base_config_path = "tuned_configs/multi_lt_transship/fine_tuned_final.json"
change_dirs = [
    "tuned_configs/multi_lt_transship/env_test/reactive_tf",
    "tuned_configs/multi_lt_transship_mechanism/env_test/shanshu_demand",
    "tuned_configs/multi_lt_transship/technique_test/share_param",
]
cmd_format_multi = "python train.py --load_config {} --load_change_config {} --exp_name {}"

cmd_list_multi = []

import itertools

# Get all json files and their exp names from each directory
dir_files = {}
for dir_path in change_dirs:
    dir_files[dir_path] = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file).replace("\\", "/")
                exp_name = json.load(open(file_path, "r", encoding='utf-8'))["main_args"]["exp_name"]
                dir_files[dir_path].append((file_path, exp_name))

# Generate combinations using exactly k paths where k = len(change_dirs)
files_from_dirs = []
exp_names = []
for dir_path in change_dirs:
    files_from_dirs.append([f[0] for f in dir_files[dir_path]])
    exp_names.append([f[1] for f in dir_files[dir_path]])
    
# Generate all combinations of files from each directory
for file_paths_combo in itertools.product(*files_from_dirs):
    combined_change_config = ",".join(file_paths_combo)
    combined_exp_name = "_".join([exp_names[i][files_from_dirs[i].index(file_paths_combo[i])]
                                for i in range(len(file_paths_combo))])
    
    cmd_list_multi.append(cmd_format_multi.format(
        base_config_path.replace("\\", "/"), 
        combined_change_config,
        combined_exp_name
    ))

# 存储为shell脚本
with open("caogao_run_train_multi.sh", "w") as f:
    for cmd in cmd_list_multi:
        f.write(cmd + "\n")

