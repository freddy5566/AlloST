from typing import Dict

import argparse

def read_phone_dictionary(dict_path: str) -> Dict[str, str]:
    phone_mapping = {}
    base = ord("\U00013000")

    with open(dict_path, "r") as phone_dict:
        lines = phone_dict.readlines()

        for line in lines:
            line = line.strip().split()
            phone, number = line[0], line[1]

            if phone == "<unk>":
                continue
            
            phone_mapping[phone] = chr(base + int(number))

    return phone_mapping

def map_phone_sequence(phone_sequence: str, phone_dict: Dict[str, str]) -> str:
    phone_sequence = phone_sequence.split(" ")
    transformed_phone_sequence = ""

    for phone in phone_sequence:
        transformed_phone_sequence += phone_dict.get(phone, "<unk>")
            
    return transformed_phone_sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phone-dict",
        type=str,
        help="path of phone dictionary",
        required=True,
    )
    parser.add_argument(
        "--phone-input",
        type=str,
        help="phone sequence of training sound",
        required=True,
    )

    args = parser.parse_args()

    phone_dict = read_phone_dictionary(args.phone_dict)

    with open(args.phone_input, "r") as phone_input:
        lines = phone_input.readlines()

        for line in lines:
            line = line.strip()

            phone = map_phone_sequence(line, phone_dict=phone_dict)

            print(phone)