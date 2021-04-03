import tempfile
import argparse

import sentencepiece as spm

from phone_mapping import read_phone_dictionary, map_phone_sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phone-path",
        type=str,
        help="path of phone sequence",
        required=True,
    )
    parser.add_argument(
        "--phone-dict",
        type=str,
        help="path of phone dictionary",
        required=True,
    )
    parser.add_argument(
        "--bpe-model",
        type=str,
        help="path of bpe model",
        required=True,
    )
    parser.add_argument(
        "--dropout-prob",
        type=float,
        help="BPE-dropout probability",
        default=0.0,
    )

    args = parser.parse_args()

    phone_dict = read_phone_dictionary(args.phone_dict)
    
    s = spm.SentencePieceProcessor(model_file=args.bpe_model)
    
    with open(args.phone_path, "r") as phone_sequence:
        phone_lines = phone_sequence.readlines()

        for phone_line in phone_lines:
            phone_line = phone_line.strip().split()
            
            uid, phone = phone_line[0], " ".join(phone_line[1:])
            transformer_phone = map_phone_sequence(phone, phone_dict) + "\n"

            bpe_phone = " ".join(s.encode(transformer_phone, out_type=str, enable_sampling=True, alpha=args.dropout_prob))

            print(f"{uid} {bpe_phone}")