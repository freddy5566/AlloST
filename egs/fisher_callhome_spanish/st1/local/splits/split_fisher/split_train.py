#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Academia Sinica (Freddy CHENG)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import subprocess

from argparse import ArgumentParser
from typing import Callable
from enum import Enum

def setup_arguments() -> ArgumentParser:
  parser = ArgumentParser()
  parser.add_argument(
    '--speech-path',
    help='speech corpus (LDC2010S01) path',
    type=str,
    required=True,
  )

  return parser

class TrainOption(Enum):
  ALL = 576000 # 160 * 60 * 60, 160 hr
  MID = 144000 # 40 * 60 * 60, 40 hr
  LOW = 72000 # 20 * 60 * 60, 20 hr

def all_strategy(index: int) -> bool:
  return True

def mid_strategy(index: int) -> bool:
  return index % 4 == 0

def low_strategy(index: int) -> bool:
  return index % 8 == 0

def split_data(
  option: TrainOption,
  index_strategy: Callable[[int], bool],
  corpus_path: str,
):
  count = 0
  
  train_filename = f'train_{option.name}'.lower()
  with open('train', 'r') as train_file, \
      open(train_filename, 'a+') as new_train_file:
    lines = train_file.readlines()
    for index, line in enumerate(lines):
      if index_strategy(index):
        filename = line.strip()
        path = corpus_path + '/' + filename
        process = subprocess.run(['sox', '--i', '-D', path],
                                  stdout=subprocess.PIPE,
                                  universal_newlines=True)
        duration = float(process.stdout.strip())
        count += duration

        if count < option.value:
          new_train_file.write(f'{filename}\n')
        else:
          break

    total_hour = count / 3600
    print(f'split {option.name} done; {total_hour:.2f} hr')

if __name__ == '__main__':
  args = setup_arguments().parse_args()
  corpus_path = args.speech_path

  split_data(TrainOption.ALL, all_strategy, corpus_path)
  split_data(TrainOption.MID, mid_strategy, corpus_path)
  split_data(TrainOption.LOW, low_strategy, corpus_path)