"""Train an algorithm."""
import argparse
import json
from harl.utils.configs_tools import get_defaults_yaml_args, update_args, t_or_f, deep_update
import sys
import os

def is_debug():
    return sys.gettrace() is not None

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="multi_lt_transship",
        choices=[
            "multi_lt_transship",
            "multi_lt_transship_mechanism",
        ],
        help="Environment name. Choose from: multi_lt_transship, multi_lt_transship_mechanism",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="tuned_configs/multi_lt_transship/fine_tuned_final.json",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--load_change_config",
        type=str,
        default="tuned_configs/multi_lt_transship_mechanism/fine_tuned_final.json",
        help="在load_config的基础上，再次修改config。多个配置文件路径用逗号分隔。",
    )

    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        if args["load_change_config"] != "":
            change_config_paths = [path.strip() for path in args["load_change_config"].split(',')]
            for change_config_path in change_config_paths:
                if os.path.exists(change_config_path):
                    with open(change_config_path, encoding="utf-8") as file:
                        change_config = json.load(file)
                    all_config = deep_update(all_config, change_config)
                else:
                    print(f"Warning: Config file {change_config_path} not found. Skipping.")
        print(all_config)
        # 更新args,除了exp_name
        exp_name = args["exp_name"]
        args = deep_update(args, all_config["main_args"])
        args["exp_name"] = exp_name
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
        if "algo_mechanism" in all_config["main_args"]:
            args["algo_mechanism"] = all_config["main_args"]["algo_mechanism"]
            algo_mechanism_args = all_config["algo_mechanism_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line

    # 如果是multi_lt_transship_mechanism, 需要在model参数中加入num_agents
    if args["env"] == "multi_lt_transship_mechanism":
        algo_mechanism_args["model"]["num_agents"] = env_args["num_agents"]

    # 识别是否在debug
    if is_debug():
        algo_args["train"]["n_rollout_threads"] = 2
        algo_args["train"]["eval_interval"] = 1
        algo_args["eval"]["n_eval_rollout_threads"] = 2
        if "algo_mechanism" in args:
            algo_mechanism_args["train"]["n_rollout_threads"] = 2
            algo_mechanism_args["train"]["warmup_steps"] = 1100
            algo_mechanism_args["eval"]["n_rollout_threads"] = 2

    # start training
    from harl.runners import RUNNER_REGISTRY

    if "algo_mechanism" in args:
        runner = RUNNER_REGISTRY["mix"](args, algo_args, env_args, algo_mechanism_args)
    else:
        runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
