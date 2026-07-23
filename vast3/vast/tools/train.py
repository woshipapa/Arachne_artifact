import argparse
from vast.train import launch_from_config


def parse_args():
    parser = argparse.ArgumentParser(description="Task")
    parser.add_argument("config", type=str, default="", help="path to config file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    launch_from_config(args.config)
