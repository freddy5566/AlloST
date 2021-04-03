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

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--phone-path", type=str, help="phone sequence file path", required=True)
    parser.add_argument("--wav-scp-path", type=str, help="wav.scp file path", required=True)
    parser.add_argument("--number-of-worker", type=int, help="number of worker for multi processing", default=8)
    parser.add_argument("--phone-system", type=str, choices=["allophone", "but"], help="phone recognition system", default="allophone")
    parser.add_argument("--dataset", type=str, help="dataset, train_sp/fisher_dev/fisher_dev2/fisher_test", required=True)

    return parser

def read_wav_scp(wav_scp_path: str) -> Dict[str, str]:
    """read wav.scp and turn it into dict of filename <-> path / command"""
    with open(wav_scp_path, "r", encoding="utf-8") as wav_scp:
        wav_scp_lines = wav_scp.readlines()

        wav = {}

    for wav_scp_line in wav_scp_lines:
        wav_scp_line = wav_scp_line.strip().split()
        filename = wav_scp_line[0]
        command = " ".join(wav_scp_line[1:])
        wav[filename] = command

    return wav

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

def recognize_phone(phone_system:str, phone_path: str, wav: Dict[str, str], number_of_worker:int):
    """read the segment file from given path, and recognize phone sequences"""
    phones, segments = [], []
    
    recognize_phone_worker = allophone_recognize_phone_worker if phone_system == 'allophone' else but_recognize_phone_worker

    with multiprocessing.Pool(number_of_worker) as pool:
        index = 0
        
        for uid in wav:
            segments.append((uid, index))
            phones.append('tmp')
            index += 1
        
        results = pool.map_async(recognize_phone_worker, segments).get()
        
        pool.close()
        pool.join()

    with open(phone_path, 'a+', encoding='utf-8') as phone_file:
        for result in results:
            uid, phone, index = result[0], result[1], int(result[2])
            phones[index] = f'{uid} {phone}\n'

        phone_file.writelines(phones)

def sph2wav(wav: Dict[str, str], phone_system:str, number_of_worker: int):
    """turn sph files into wav format"""
    args = list(map(lambda i: (i, phone_system), wav.items()))

    with multiprocessing.Pool(number_of_worker) as pool:
        args = list(map(lambda i: (i, phone_system), wav.items()))

        pool.starmap_async(sph2wav_worker, args)

        pool.close()
        pool.join()

def sph2wav_worker(item: Tuple[str, str], phone_system: str):
    filename, sph2wav_command = item

    if not os.path.exists(f'./{phone_system}/{filename}.wav'):
        sph2wav_command = sph2wav_command.strip()
        sph2wav_commands = list(map(lambda command: command.strip().lstrip().split(), sph2wav_command.split('|')))

        speed = sph2wav_commands[0][-1]
        wav_file_location = sph2wav_commands[0][3]

        uid = sph2wav_commands[0]

        sp_command = f'sox {wav_file_location} {phone_system}/{filename}.wav speed {speed}'.split()

        sp_process = subprocess.run(sp_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        sp_out = sp_process.stdout.decode('utf-8').strip()
        
        if sp_out:
            print(f'[sph2wav][{filename}]: {sp_out}') 

if __name__ == "__main__":
    args = get_parser().parse_args()

    if not os.path.exists(args.phone_system):
        os.makedirs(args.phone_system)

    wav = read_wav_scp(wav_scp_path=args.wav_scp_path)

    # if args.dataset == "taigi_train_sp.tai":
    #     sph2wav(wav=wav, phone_system=args.phone_system, number_of_worker=args.number_of_worker)     
    # else:
    #     # copy taigi speech to allophone folder
    #     for uid in wav:
    #         taigi_speech_path = wav[uid]
    #         os.system(f"cp {taigi_speech_path} {args.phone_system}/{uid}.wav")

    recognize_phone(
        phone_system='allophone' if args.phone_system == 'allophone' else 'but',
        wav=wav,
        phone_path=args.phone_path,
        number_of_worker=args.number_of_worker,
    )