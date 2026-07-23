import argparse
from teleai_data_tool.logger import logger
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Task")
    parser.add_argument("--config", type=str, default="", help="path to config file")
    parser.add_argument(
        "--runners", type=str, default="", help="runners of executing task"
    )
    args = parser.parse_args()
    return args


def run_tasks(config_path, runners):
    from vast.train import Tester, Trainer, load_config, utils

    config = load_config(config_path)
    project_dir = config.get("project_dir", None)
    if project_dir is None:
        project_name = os.path.splitext(os.path.basename(config_path))[0]
        project_dir = os.path.join(os.getcwd(), "work_dirs", project_name)
        config["project_dir"] = project_dir
    logger.info(f"project dir is {project_dir}")
    for runner in runners:
        runner = utils.import_function(runner)
        runner = runner.load(config)
        runner.print(config)
        if isinstance(runner, Trainer):
            runner.save_config(config)
            if config.train.get("resume", False):
                runner.resume()
            runner.train()
        elif isinstance(runner, Tester):
            runner.test()
        else:
            assert False


def main():
    args = parse_args()
    runners = args.runners.split(",")
    run_tasks(args.config, runners)


if __name__ == "__main__":
    main()
