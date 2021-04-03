#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Academia Sinica (YAO-FEI, CHENG)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from allosaurus.app import read_recognizer

from typing import Dict, Tuple, List

import multiprocessing
import subprocess
import argparse
import os

def get_parser():
  parser = argparse.ArgumentParser(
    description='produce phone from given text file',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,  
  )

  parser.add_argument('--phone-path', type=str, help='phone sequence file path', required=True)
  parser.add_argument('--wav-scp-path', type=str, help='wav.scp file path', required=True)
  parser.add_argument('--segments-path', type=str, help='segments file path', required=True)
  parser.add_argument('--number-of-worker', type=int, help='number of worker for multi processing', default=8)
  parser.add_argument('--phone-system', type=str, choices=['allophone', 'but'], help='phone recognition system', default='allophone')
  parser.add_argument('--dataset', type=str, help='dataset, train_sp/fisher_dev/fisher_dev2/fisher_test', required=True)
  
  return parser

def read_wav_scp(wav_scp_path: str) -> Dict[str, str]:
  """read wav.scp and turn it into dict of filename <-> command"""
  with open(wav_scp_path, 'r', encoding='utf-8') as wav_scp:
    wav_scp_lines = wav_scp.readlines()
    
    wav = {}
    
    for wav_scp_line in wav_scp_lines:
      wav_scp_line = wav_scp_line.strip().split()
      filename = wav_scp_line[0]
      command = ' '.join(wav_scp_line[1:])
      wav[filename] = command

  return wav

def sph2wav_worker(item: Tuple[str, str], phone_system: str):
  """turn sph file into wav format"""
  filename, sph2wav_command = item
  
  if not os.path.exists(f'./{phone_system}/{filename}.wav'):
    sph2wav_command = sph2wav_command.strip()
    sph2wav_commands = list(map(lambda command: command.strip().lstrip().split(), sph2wav_command.split('|')))

    sample = sph2wav_commands[1][-2]
    
    resample_command = f'sox -G -R -t wav - {phone_system}/{filename}.wav rate {sample} dither'.split()

    sph2wav_process = subprocess.Popen(sph2wav_commands[0], stdout=subprocess.PIPE)
    resample_process = subprocess.run(resample_command, stdin=sph2wav_process.stdout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    resample_out = resample_process.stdout.decode('utf-8').strip()
    
    if resample_out:
      print(f'[sph2wav][{filename}]: {resample_out}')

def sph2wav_sp_worker(item: Tuple[str, str], phone_system: str):
  """turn sph file into wav format"""
  filename, sph2wav_command = item
  
  if not os.path.exists(f'./{phone_system}/{filename}.wav'):
    sph2wav_command = sph2wav_command.strip()
    sph2wav_commands = list(map(lambda command: command.strip().lstrip().split(), sph2wav_command.split('|')))

    sample = sph2wav_commands[1][-2]
    speed = sph2wav_commands[-2][-1]
    uid = '-'.join(filename.split('-')[1:])

    resample_command = f'sox -G -R -t wav - {phone_system}/{uid}.wav rate {sample} dither'.split()
    sp_command = f'sox {phone_system}/{uid}.wav {phone_system}/{filename}.wav speed {speed}'.split()

    sph2wav_process = subprocess.Popen(sph2wav_commands[0], stdout=subprocess.PIPE)
    resample_process = subprocess.run(resample_command, stdin=sph2wav_process.stdout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    if not os.path.exists(f'./{phone_system}/{uid}.wav'):
      resample_process = subprocess.run(resample_command, stdin=sph2wav_process.stdout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
      resample_out = resample_process.stdout.decode('utf-8').strip()

      if resample_out:
        print(f'[sph2wav][{filename}]: {resample_out}')

      sp_process = subprocess.run(sp_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    else:
      sp_process = subprocess.run(sp_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    sp_out = sp_process.stdout.decode('utf-8').strip()
    
    if sp_out:
      print(f'[sph2wav][{filename}]: {sp_out}')


def sph2wav(dataset: str, wav: Dict[str, str], phone_system:str, number_of_worker: int):
  """turn sph files into wav format"""
  with multiprocessing.Pool(number_of_worker) as pool:
    args = list(map(lambda i: (i, phone_system), wav.items()))
    
    if dataset == "train_sp.es":
      pool.starmap_async(sph2wav_sp_worker, args)
    else:
      pool.starmap_async(sph2wav_worker, args)

    pool.close()
    pool.join()

def trim_wav_worker(segment_line: str, phone_system: str):
  """trim the wav file"""
  segment_line = segment_line.strip().split()
  uid = segment_line[0]
  filename = segment_line[1]
  start = segment_line[2]
  end = segment_line[3]

  duration = float(end) - float(start)

  trim_command =  f'ffmpeg -loglevel panic -nostdin ' + \
                  f'-i {phone_system}/{filename}.wav -ss {start} ' + \
                  f'-to {end} -c copy {phone_system}/{uid}.wav'
  
  trim_process = subprocess.run(trim_command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

  trim_output = trim_process.stdout.decode('utf-8').strip()
  
  if trim_output:
    print(f'[trim][{uid}]: {trim_output}')

def trim_wav(segments_path: str, number_of_worker: int, phone_system: str):
  """trim wav files"""
  with open(segments_path, 'r', encoding='utf-8') as segments_file:
    segment_lines = segments_file.readlines()
    args = list(map(lambda s: (s, phone_system), segment_lines))

    with multiprocessing.Pool(number_of_worker) as pool:
      pool.starmap_async(trim_wav_worker, args)

      pool.close()
      pool.join()

def allophone_recognize_phone_worker(item: Tuple[str, int]) -> List[str]:
  """recognize phone sequence from given file by allophone"""
  uid, index = item
  
  model = read_recognizer()
  try:
    phone = model.recognize(f'./allophone/{uid}.wav')
    if len(phone) == 0:
      phone = 'sil'

  except Exception as e:
    phone = 'sil'
    print(f'[phone][{uid}] {e}')

  return [uid, phone, str(index)]

def but_recognize_phone_worker(item: Tuple[str, int]) -> List[str]:
  """recognize phone sequence from given file by but"""
  uid, index = item
  current_path = os.getcwd()
  
  but_command = f'docker run -v {current_path}/but:/usr/src/but phnrec ' + \
                f'./PhnRec/phnrec -v -c ./PhnRec/PHN_CZ_SPDAT_LCRC_N1500 ' + \
                f'-s wf -t str -w lin16 ' + \
                f'-i /usr/src/but/{uid}.wav ' + \
                f'-o /usr/src/but/{uid}.lab'
  
  but_process = subprocess.run(but_command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  but_output = but_process.stdout.decode('utf-8').strip()
  
  if but_output:
    print(f'[but][{uid}]: {but_output}')
  
  phones = []
  segments = []
  with open(f'./but/{uid}.lab', 'r', encoding='utf-8') as lab:
    lab_lines = lab.readlines()

    for lab_line in lab_lines:
      lab_line = lab_line.strip().split()
      start = float(lab_line[0]) * 1e-4
      end = float(lab_line[1]) * 1e-4
      phone = lab_line[2]
      current_phone = phone
      
      segments.append({'start': start, 'end': end, 'phone': phone})

    # from zero to duration, step every single 10ms
    for start_of_frame in range(0, 1220 - 25 + 1, 10):
      end_of_frame = start_of_frame + 25
      frame_phones = []
      for segment in segments:
        # frame is exactly inside this segment
        if segment['start'] >= start_of_frame and segment['end'] <= end_of_frame:
          frame_phones.append(segment['phone'])
        # segment is between this frame and next frame
        elif segment['start'] < end_of_frame and segment['end'] > end_of_frame:
          ratio = (end_of_frame - segment['start']) / 25
          if ratio > 0.5:
            frame_phones.append(segment['phone'])
        # # segment is between this frame and last frame
        elif segment['start'] < start_of_frame and segment['end'] > start_of_frame:
          ratio = (segment['end'] - start_of_frame) / 25
          if ratio > 0.5:
            frame_phones.append(segment['phone'])

      if len(frame_phones) == 0:
        frame_phones.append('sil')
      phones.append(''.join(frame_phones))

  return [uid, ' '.join(phones), str(index)]

def recognize_phone(phone_system:str, phone_path: str, segments_path: str, number_of_worker:int):
  """read the segment file from given path, and recognize phone sequences"""
  phones = []
  segments = []

  recognize_phone_worker = allophone_recognize_phone_worker if phone_system == 'allophone' else but_recognize_phone_worker

  with multiprocessing.Pool(number_of_worker) as pool:
    with open(segments_path, 'r', encoding='utf-8') as segments_file:
      segment_lines = segments_file.readlines()

      for index, segment_line in enumerate(segment_lines):
        segment_line = segment_line.strip().split()
        uid = segment_line[0]

        segments.append((uid, index))
        phones.append('tmp')

    results = pool.map_async(recognize_phone_worker, segments).get()
    pool.close()
    pool.join()

    with open(phone_path, 'a+', encoding='utf-8') as phone_file:
      for result in results:
        uid, phone, index = result[0], result[1], int(result[2])
        phones[index] = f'{uid} {phone}\n'

      phone_file.writelines(phones)

if __name__ == '__main__':
  args = get_parser().parse_args()

  if not os.path.exists(args.phone_system):
    os.makedirs(args.phone_system)

  wav = read_wav_scp(wav_scp_path=args.wav_scp_path)  
  sph2wav(dataset=args.dataset, wav=wav, phone_system=args.phone_system, number_of_worker=args.number_of_worker)
  trim_wav(segments_path=args.segments_path, phone_system=args.phone_system, number_of_worker=args.number_of_worker)

  recognize_phone(
    phone_system='allophone' if args.phone_system == 'allophone' else 'but',
    phone_path=args.phone_path,
    segments_path=args.segments_path,
    number_of_worker=args.number_of_worker,
  )