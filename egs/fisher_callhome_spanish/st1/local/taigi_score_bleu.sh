#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. ./path.sh

nlsyms=""
bpe=""
bpemodel=""
filter=""
case=lc
set=""

. utils/parse_options.sh

if [ $# -lt 2 ]; then
    echo "Usage: $0 <decode-dir> <dict-tgt> <dict-src>";
    exit 1;
fi

dir=$1
dic_tgt=$2
dic_src=$3

concatjson.py ${dir}/data.*.json > ${dir}/data.json
json2trn_mt.py ${dir}/data.json ${dic_tgt} --refs ${dir}/ref.trn.org \
    --hyps ${dir}/hyp.trn.org --srcs ${dir}/src.trn.org --dict-src ${dic_src}
if [ ! -z ${set} ] && [ -f ${dir}/data_ref1.json ]; then
    json2trn_mt.py ${dir}/data_ref1.json ${dic_tgt} --refs ${dir}/ref1.trn.org
    json2trn_mt.py ${dir}/data_ref2.json ${dic_tgt} --refs ${dir}/ref2.trn.org
    json2trn_mt.py ${dir}/data_ref3.json ${dic_tgt} --refs ${dir}/ref3.trn.org
fi

# remove uttterance id
perl -pe 's/\([^\)]+\)//g;' ${dir}/ref.trn.org > ${dir}/ref.trn
perl -pe 's/\([^\)]+\)//g;' ${dir}/hyp.trn.org > ${dir}/hyp.trn
perl -pe 's/\([^\)]+\)//g;' ${dir}/src.trn.org > ${dir}/src.trn
if [ ! -z ${set} ] && [ -f ${dir}/data_ref1.json ]; then
    perl -pe 's/\([^\)]+\)//g;' ${dir}/ref1.trn.org > ${dir}/ref1.trn
    perl -pe 's/\([^\)]+\)//g;' ${dir}/ref2.trn.org > ${dir}/ref2.trn
    perl -pe 's/\([^\)]+\)//g;' ${dir}/ref3.trn.org > ${dir}/ref3.trn
fi

if [ ! -z ${bpemodel} ]; then
    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref.trn | sed -e "s/▁/ /g" > ${dir}/ref.wrd.trn
    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
    spm_decode --model=${bpemodel} --input_format=piece < ${dir}/src.trn | sed -e "s/▁/ /g" > ${dir}/src.wrd.trn
    if [ ! -z ${set} ] && [ -f ${dir}/data_ref1.json ]; then
        spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref1.trn | sed -e "s/▁/ /g" > ${dir}/ref1.wrd.trn
        spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref2.trn | sed -e "s/▁/ /g" > ${dir}/ref2.wrd.trn
        spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref3.trn | sed -e "s/▁/ /g" > ${dir}/ref3.wrd.trn
    fi
else
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/ref.trn > ${dir}/ref.wrd.trn
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
    sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/src.trn > ${dir}/src.wrd.trn
    if [ ! -z ${set} ] && [ -f ${dir}/data_ref1.json ]; then
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/ref1.trn >> ${dir}/ref1.wrd.trn
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/ref2.trn >> ${dir}/ref2.wrd.trn
        sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g" ${dir}/ref3.trn >> ${dir}/ref3.wrd.trn
    fi
fi

# 1 reference
echo "1-ref BLEU"
multi-bleu-detok.perl -lc ${dir}/ref.wrd.trn < ${dir}/hyp.wrd.trn >> ${dir}/result.txt

echo "write a case-insensitive BLEU result in ${dir}/result.lc.txt"
cat ${dir}/result.txt

# TODO(hirofumi): add TER & METEOR metrics here
