#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Academia Sinica (YAO-FEI, CHENG)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from argparse import ArgumentParser
from typing import List, Callable
from pathlib import Path

def setup_arguments() -> ArgumentParser:
  parser = ArgumentParser()
  parser.add_argument(
    '--type',
    help='data set type, all, mid, low',
    default='all',
    nargs='?',
    choices=['all', 'mid', 'low'],
  )

  parser.add_argument(
    '--file',
    help='which file(s) will be extract',
    default='all',
    nargs='?',
    choices=['all', 'en', 'es']
  )

  return parser

def get_splits_ids(data_type: str) -> List[str]:
  """get all of ids from train_{type}, e.g., train_low
    splitted data is comming from split_train.py
  """
  split_data_path = (Path(__file__).parent / f'./splits/split_fisher/train_{data_type}').resolve()
  
  split_ids = [] # e.g., 20050908_182943_22_fsp
  with open(split_data_path) as split_file:
    split_lines = split_file.readlines()
    for split_line in split_lines:
      split_line = split_line.strip()
      split_id = split_line.split('.')[0]
      split_ids.append(split_id)

  return split_ids

def extract_splitted_file(
  original_file_path: Path,
  new_file_path: Path,
  split_ids: List[str],
):
  """extract needed line from given ids"""
  with open(original_file_path, 'r', encoding='utf-8') as original_file, \
    open(new_file_path, 'a+', encoding='utf-8') as new_file:

    original_lines = original_file.readlines()

    for original_line in original_lines:
      original_line = original_line.strip()
      split_id = original_line.split(' ')[0].split('-')[0]

      if split_id in split_ids:
        new_file.write(f'{original_line}\n')

def extract_transcription(
  original_text_path: Path,
  original_transcription_path: Path,
  new_transcription_path: Path,
  split_ids: List[str],
):
  """extract es transcription from mapping given text and es transcription"""
  with open(original_text_path, 'r', encoding='utf-8') as original_text, \
    open(original_transcription_path, 'r', encoding='utf-8') as original_transcription, \
    open(new_transcription_path, 'a+', encoding='utf-8') as new_transcription:

    text_lines = original_text.readlines()
    original_transcription_lines = original_transcription.readlines()

    for line_no, text_line in enumerate(text_lines):
      text_line = text_line.strip()
      split_id = text_line.split(' ')[0].split('-')[0]

      if split_id in split_ids:
        original_transcription_line = original_transcription_lines[line_no].strip()
        new_transcription.write(f'{original_transcription_line}\n')

if __name__ == '__main__':
  args = setup_arguments().parse_args()
  
  train_root = '../data/fisher_train'
  split_train_root = f'../data/fisher_train_{args.type}'

  # split ids from split.{type}
  split_ids = get_splits_ids(data_type=args.type)

  if args.file == 'all':
    train_type_path = (Path(__file__).parent / split_train_root).resolve()
    Path(train_type_path).mkdir(parents=True, exist_ok=True)

    need_copy_files = [
      'reco2file_and_channel',
      'segments',
      'spk2gender',
      'spk2utt',
      'text',
      'utt2spk',
      'wav.scp',
    ]

    for filename in need_copy_files:
      original_file_path = (Path(__file__).parent / f'{train_root}/{filename}').resolve()
      new_path = (Path(__file__).parent / f'{split_train_root}/{filename}').resolve()
      
      extract_splitted_file(
        original_file_path=original_file_path,
        new_file_path=new_path,
        split_ids=split_ids,
      )

  elif args.file == 'es':
    # extract lines from es.joshua.org.tmp
    text_path = (Path(__file__).parent / f'{train_root}/text').resolve()
    original_es_path = (Path(__file__).parent / f'{train_root}/es.joshua.org.tmp').resolve()
    new_es_path = (Path(__file__).parent / f'{split_train_root}/es.joshua.org').resolve()

    extract_transcription(
      original_text_path=text_path,
      original_transcription_path=original_es_path,
      new_transcription_path=new_es_path,
      split_ids=split_ids,
    )

  elif args.file == 'en':
    # extract lines from en.norm.tc.tmp
    text_path = (Path(__file__).parent / f'{train_root}/text.org').resolve()
    original_en_path = (Path(__file__).parent / f'{train_root}/en.norm.tc.tmp').resolve()
    new_en_path = (Path(__file__).parent / f'{train_root}/en.norm.tc').resolve()

    extract_transcription(
      original_text_path=text_path,
      original_transcription_path=original_en_path,
      new_transcription_path=new_en_path,
      split_ids=split_ids,
    )
  else:
    print(f'error on extract data')