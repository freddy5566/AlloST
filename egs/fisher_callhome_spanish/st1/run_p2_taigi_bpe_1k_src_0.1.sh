#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch # chainer or pytorch
stage=5         # start from 0 if you need to start from data preparation
stop_stage=5
ngpu=1          # number of gpus ("0" uses cpu, otherwise use gpu)
nj=4            # number of parallel jobs for decoding
debugmode=1
dumpdir=dump_taigi_bpe_1k_src_0.1    # directory to dump full features
N=0             # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0       # verbose option
resume=         # Resume the training from snapshot
seed=1          # seed to generate random number
# feature configuration
do_delta=false

preprocess_config=
train_config=conf/tuning/train_dual_encoder_conformer.yaml
decode_config=conf/decode.yaml

trans_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# model average realted (only for transformer)
n_average=5                  # the number of ST models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ST models will be averaged.
                             # if false, the last `n_average` ST models will be averaged.
metric=bleu                  # loss/acc/bleu

# pre-training related
asr_model=
mt_model=

# preprocessing related
src_case=lc.rm
tgt_case=lc.rm

taigi_corpus=/mnt/md0/user_jamfly/CORPUS/Taigi

# wd: word level
# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.


# bpemode (unigram or bpe)
nbpe=10000
bpemode=bpe
phone_nbpe=1000
bpe_dropout=0
phone_bpe_dropout=0.1
# NOTE: nbpe=53 means character-level ST (lc.rm)
# NOTE: nbpe=66 means character-level ST (lc)
# NOTE: nbpe=98 means character-level ST (tc)

# set training data split type, all, mid, low
# all is 160 hr, mid 40 hr, low 20 hr, respectively
split_type="mid"

# phone recognition system
phone_system="allophone"

# exp tag
tag="taigi_p2_bpe_1k_src_0.1" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=taigi_train_sp.zh
train_set_prefix=taigi_train_sp
train_dev=taigi_train_dev.zh
trans_set="taigi_test.zh"
train_nj=8

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data Preparation"
    local/data_prep.sh ${taigi_corpus}
fi


feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    # fbankdir=fbank
    # # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    # for x in taigi_train taigi_train_dev taigi_test; do
    #     steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${train_nj} --write_utt2num_frames true \
    #         data/${x} exp/make_fbank/${x} ${fbankdir}
    # done

    # # speed-perturbed. data/${train_set_ori} is the orignal and data/${train_set} is the augmented
    # utils/perturb_data_dir_speed.sh 0.9 data/taigi_train data/temp1
    # utils/perturb_data_dir_speed.sh 1.0 data/taigi_train data/temp2
    # utils/perturb_data_dir_speed.sh 1.1 data/taigi_train data/temp3
    # utils/combine_data.sh --extra-files utt2uniq data/taigi_train_sp data/temp1 data/temp2 data/temp3
    
    # rm -r data/temp1 data/temp2 data/temp3
    # steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${train_nj} --write_utt2num_frames true \
    #     data/taigi_train_sp exp/make_fbank/taigi_train_sp ${fbankdir}

    # for lang in tai zh; do
    #     awk -v p="sp0.9-" '{printf("%s %s%s\n", $1, p, $1);}' data/taigi_train/utt2spk > data/taigi_train_sp/utt_map
    #     utils/apply_map.pl -f 1 data/taigi_train_sp/utt_map < data/taigi_train/text.${lang} > data/taigi_train_sp/text.${lang}
       
    #     awk -v p="sp1.0-" '{printf("%s %s%s\n", $1, p, $1);}' data/taigi_train/utt2spk > data/taigi_train_sp/utt_map
    #     utils/apply_map.pl -f 1 data/taigi_train_sp/utt_map < data/taigi_train/text.${lang} >> data/taigi_train_sp/text.${lang}
       
    #     awk -v p="sp1.1-" '{printf("%s %s%s\n", $1, p, $1);}' data/taigi_train/utt2spk > data/taigi_train_sp/utt_map
    #     utils/apply_map.pl -f 1 data/taigi_train_sp/utt_map < data/taigi_train/text.${lang} >> data/taigi_train_sp/text.${lang}

    # done
    # utils/fix_data_dir.sh data/taigi_train_sp
    # utils/validate_data_dir.sh --no-text data/taigi_train_sp

    # # Divide into source and target languages
    # for x in ${train_set_prefix} taigi_train_dev taigi_test; do
    #     local/divide_taigi_lang.sh ${x}
    # done

    # for x in ${train_set_prefix} taigi_train_dev; do
    #     # remove utt having more than 3000 frames
    #     # remove utt having more than 400 characters
    #     for lang in zh tai; do
    #         remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${x}.${lang} data/${x}.${lang}.tmp
    #     done
    #     # Match the number of utterances between source and target languages
    #     # extract commocn lines
    #     cut -f 1 -d " " data/${x}.tai.tmp/text > data/${x}.zh.tmp/reclist1
    #     cut -f 1 -d " " data/${x}.zh.tmp/text > data/${x}.zh.tmp/reclist2
    #     comm -12 data/${x}.zh.tmp/reclist1 data/${x}.zh.tmp/reclist2 > data/${x}.zh.tmp/reclist

    #     for lang in tai zh; do
    #         reduce_data_dir.sh data/${x}.${lang}.tmp data/${x}.zh.tmp/reclist data/${x}.${lang}
    #         utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${x}.${lang}
    #     done
    #     rm -rf data/${x}.*.tmp
    # done

    # # compute global CMVN
    # compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
    
    # # dump features for training
    # dump.sh --cmd "$train_cmd" --nj ${train_nj} --do_delta $do_delta \
    #     data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_set} ${feat_tr_dir}
    # dump.sh --cmd "$train_cmd" --nj ${train_nj} --do_delta $do_delta \
    #     data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_dev} ${feat_dt_dir}
    # for ttask in ${trans_set}; do
    #     feat_trans_dir=${dumpdir}/${ttask}/delta${do_delta}; mkdir -p ${feat_trans_dir}
    #     dump.sh --cmd "$train_cmd" --nj ${train_nj} --do_delta $do_delta \
    #         data/${ttask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/trans/${ttask} \
    #         ${feat_trans_dir}
    # done

    echo "recognizing phone...."
    for dataset in taigi_train_dev.tai taigi_train_sp.tai taigi_test.tai; do
        python local/taigi_phone_recognition.py \
          --phone-path data_taigi_bpe_1k_src_0.1/${dataset}/phone \
          --wav-scp-path data_taigi_bpe_1k_src_0.1/${dataset}/wav.scp \
          --dataset ${dataset} \
          --phone-system ${phone_system} \
          --number-of-worker ${nj} > data_taigi_bpe_1k_src_0.1/${dataset}/phone.log
    done
    # rm -rf ${phone_system}
fi

dict=data_taigi_bpe_1k_src_0.1/lang_1spm/${train_set}_units_${tgt_case}.txt
nlsyms=data_taigi_bpe_1k_src_0.1/lang_1spm/${train_set}_non_lang_syms_${tgt_case}.txt
bpemodel=data_taigi_bpe_1k_src_0.1/lang_1spm/${train_set}_${bpemode}${nbpe}_${tgt_case}
phone_dict=data_taigi_bpe_1k_src_0.1/lang_1spm/phone.txt
phone_transform_dict=data_taigi_bpe_1k_src_0.1/lang_1spm/phone_map.txt
phone_bpemodel=data_taigi_bpe_1k_src_0.1/lang_1spm/${train_set}_${bpemode}${nbpe}_phone

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data_taigi_bpe_1k_src_0.1/lang_1spm

    echo "make a non-linguistic symbol list for all languages"
    touch ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    offset=$(wc -l < ${dict})

    grep sp1.0 data_taigi_bpe_1k_src_0.1/${train_set_prefix}.zh/text | cut -f 2- -d' ' | grep -v -e '^\s*$' > data_taigi_bpe_1k_src_0.1/lang_1spm/input.txt
    spm_train --user_defined_symbols="$(tr "\n" "," < ${nlsyms})" --input=data_taigi_bpe_1k_src_0.1/lang_1spm/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --character_coverage=1.0
    spm_encode --model=${bpemodel}.model --output_format=piece < data_taigi_bpe_1k_src_0.1/lang_1spm/input.txt | tr ' ' '\n' | sort | uniq | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict}
    wc -l ${dict}

    echo "<unk> 1" > ${phone_dict}
    phone_offset=$(wc -l < ${phone_dict})

    # phone
    grep sp1.0 data_taigi_bpe_1k_src_0.1/${train_set_prefix}.tai/phone | cut -f 2- -d' ' | grep -v -e '^\s*$' > data_taigi_bpe_1k_src_0.1/lang_1spm/phone_input.txt
    cat < data_taigi_bpe_1k_src_0.1/lang_1spm/phone_input.txt | tr ' ' '\n' | sort | uniq | awk -v offset=${phone_offset} '{print $0 " " NR+offset}' >> ${phone_dict}

    # phone bpe
    echo "<unk> 1" > ${phone_transform_dict}
    phone_transform_offset=$(wc -l < ${phone_transform_dict})

    python local/phone_mapping.py --phone-dict=${phone_dict} --phone-input=data_taigi_bpe_1k_src_0.1/lang_1spm/phone_input.txt >> data_taigi_bpe_1k_src_0.1/lang_1spm/phone_transform.txt

    spm_train --user_defined_symbols="$(tr "\n" "," < ${nlsyms})" --input=data_taigi_bpe_1k_src_0.1/lang_1spm/phone_transform.txt --vocab_size=${phone_nbpe} --model_type=${bpemode} --model_prefix=${phone_bpemodel} --input_sentence_size=100000000 --character_coverage=1.0
    spm_encode --model=${phone_bpemodel}.model --output_format=piece < data_taigi_bpe_1k_src_0.1/lang_1spm/phone_transform.txt | tr ' ' '\n' | sort | uniq | awk -v offset=${phone_transform_offset} '{print $0 " " NR+offset}' >> ${phone_transform_dict}

    echo "make json files"
    data2json.sh --nj ${train_nj} --feat ${feat_tr_dir}/feats.scp --text data_taigi_bpe_1k_src_0.1/${train_set}/text.${tgt_case} --bpecode ${bpemodel}.model --lang zh \
        data_taigi_bpe_1k_src_0.1/${train_set} ${dict} > ${feat_tr_dir}/data.${src_case}_${tgt_case}.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --text data_taigi_bpe_1k_src_0.1/${train_dev}/text.${tgt_case} --bpecode ${bpemodel}.model --lang zh \
        data_taigi_bpe_1k_src_0.1/${train_dev} ${dict} > ${feat_dt_dir}/data.${src_case}_${tgt_case}.json
    for ttask in ${trans_set}; do
        feat_trans_dir=${dumpdir}/${ttask}/delta${do_delta}
        data2json.sh --feat ${feat_trans_dir}/feats.scp --text data/${ttask}/text.${tgt_case} --bpecode ${bpemodel}.model --lang zh \
            data_taigi_bpe_1k_src_0.1/${ttask} ${dict} > ${feat_trans_dir}/data.${src_case}_${tgt_case}.json
    done

    # update json (add source references)
    for x in ${train_set} ${trans_set}; do
        feat_dir=${dumpdir}/${x}/delta${do_delta}
        data_dir=data_taigi_bpe_1k_src_0.1/$(echo ${x} | cut -f 1 -d ".").tai
        python local/phone_bpe.py --phone-path=${data_dir}/phone --phone-dict=${phone_dict} --bpe-model=${phone_bpemodel}.model >> ${data_dir}/phone_bpe
        
        update_json.sh --text ${data_dir}/text.${src_case} \
            ${feat_dir}/data.${src_case}_${tgt_case}.json ${data_dir} ${dict} "char"

        update_json.sh --text ${data_dir}/phone_bpe \
            ${feat_dir}/data.${src_case}_${tgt_case}.json ${data_dir} ${phone_transform_dict} "phn"
    done

fi

# NOTE: skip stage 3: LM Preparation

if [ -z ${tag} ]; then
    expname=${train_set}_${tgt_case}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
    if [ -n "${asr_model}" ]; then
        expname=${expname}_asrtrans
    fi
    if [ -n "${mt_model}" ]; then
        expname=${expname}_mttrans
    fi
else
    expname=${train_set}_${tgt_case}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        st_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --seed ${seed} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data.${src_case}_${tgt_case}.json \
        --valid-json ${feat_dt_dir}/data.${src_case}_${tgt_case}.json \
        --enc-init ${asr_model} \
        --dec-init ${mt_model}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
       [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]]; then
        # Average ST models
        if ${use_valbest_average}; then
            trans_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log --metric ${metric}"
        else
            trans_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${trans_model} \
            --num ${n_average}
    fi

    pids=() # initialize pids
    for ttask in ${trans_set}; do
    (
        decode_dir=decode_${ttask}_$(basename ${decode_config%.*})
        feat_trans_dir=${dumpdir}/${ttask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_trans_dir}/data.${src_case}_${tgt_case}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            st_trans.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --trans-json ${feat_trans_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${trans_model}



        local/taigi_score_bleu.sh --case ${tgt_case} --set ${ttask} --bpe ${nbpe} --bpemodel ${bpemodel}.model \
            ${expdir}/${decode_dir} ${dict}


    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
