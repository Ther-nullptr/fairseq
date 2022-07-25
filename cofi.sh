python3 fairseq_cli/hydra_train.py -m \
    --config-dir /mnt/lustre/sjtu/home/xc915/superb/wyj-fairseq/examples/cofi/config \
    --config-name cofi \
    task.data=/mnt/lustre/sjtu/home/xc915/superb/dataset/librispeech_finetuning_data/100h \
    task.label_dir=/mnt/lustre/sjtu/home/xc915/superb/dataset/librispeech_finetuning_data/100h \
    common.user_dir=/mnt/lustre/sjtu/home/xc915/superb/wyj-fairseq/examples/cofi \
    hydra.run.dir=hydra_cofi
