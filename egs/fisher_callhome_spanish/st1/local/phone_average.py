import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    phone_length = 0
    bpe_phone_length = 0

    with open(f"{args.corpus}/{args.dataset}/phone") as phone_file, \
        open(f"{args.corpus}/{args.dataset}/phone_bpe") as phone_bpe_file:
        
        phone_lines = phone_file.readlines()
        bpe_phone_lines = phone_bpe_file.readlines()

        for phone_line in phone_lines:
            phone_line = phone_line.strip()
            phone_line = phone_line.split()[1:]
            
            phone_length += len(phone_line)

        for bpe_phone_line in bpe_phone_lines:
            bpe_phone_line = bpe_phone_line.strip()
            bpe_phone_line = bpe_phone_line.split()[1:]

            bpe_phone_length += len(bpe_phone_line)

        phone_avag = phone_length / len(phone_lines)
        bpe_phone_avg = bpe_phone_length / len(bpe_phone_lines)

        print(f"phone avg: {phone_avag}")
        print(f"bpe phone avg: {bpe_phone_avg}")
    