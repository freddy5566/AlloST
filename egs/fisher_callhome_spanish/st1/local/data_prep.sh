#!/bin/bash

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. utils/parse_options.sh || exit 1;

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <Taigi-location>"
    echo "e.g.: $0 /home/mpost/data/Taigi"
    exit 1
fi

dst=$1

data=data
eval_set=$(pwd)/${data}/taigi_test
train_set=$(pwd)/${data}/taigi_train
train_dev_set=$(pwd)/${data}/taigi_train_dev

# eval set
mkdir -p ${eval_set}
awk -v taigi="${dst}" -F " " '{ if (system("test -f " taigi "/wav/" $1 ".wav") == 0) print $0 }' ${dst}/taigi_eval >> ${eval_set}/text.tai
awk -v taigi="${dst}" -F " " '{ if (system("test -f " taigi "/wav/" $1 ".wav") == 0) print $0 }' ${dst}/text_eval >> ${eval_set}/text.zh

# split train in to train and train_dev
mkdir -p ${train_set}
mkdir -p ${train_dev_set}

awk -v taigi="${dst}" -F " " 'NR <= 6000 { if (system("test -f " taigi "/wav/" $1 ".wav") == 0) print $0 }' ${dst}/taigi_train > ${train_dev_set}/text.tai
awk -v taigi="${dst}" -F " " 'NR <= 6000 { if (system("test -f " taigi "/wav/" $1 ".wav") == 0) print $0 }'  ${dst}/text_train > ${train_dev_set}/text.zh

awk -v taigi="${dst}" -F " " 'NR > 6000 { if (system("test -f " taigi "/wav/" $1 ".wav") == 0) print $0 }' ${dst}/taigi_train > ${train_set}/text.tai
awk -v taigi="${dst}" -F " " 'NR > 6000 { if (system("test -f " taigi "/wav/" $1 ".wav") == 0) print $0 }' ${dst}/text_train > ${train_set}/text.zh

for x in ${eval_set} ${train_set} ${train_dev_set}; do
    awk -v taigi="${dst}" -F " " '{ print $1 " " taigi "/wav/" $1 ".wav" }' ${x}/text.tai >> ${x}/wav.scp
    awk -F " " '{ print $1 " " $1 }' ${x}/text.tai >> ${x}/utt2spk

    utils/utt2spk_to_spk2utt.pl < ${x}/utt2spk > ${x}/spk2utt || exit 1
done

echo "$0: successfully prepared data in ${dst}"
exit 0;
