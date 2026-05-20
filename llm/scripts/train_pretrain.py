#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from mini_llm.train.pretrain import pretrain


def main():
    parser = argparse.ArgumentParser(description="Run pretraining")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    pretrain(args.config)


if __name__ == "__main__":
    main()
