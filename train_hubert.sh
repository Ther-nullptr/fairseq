python fairseq_cli/hydra_train.py \
  --config-dir /mnt/lustre/sjtu/home/xc915/superb/wyj-fairseq/examples/hubert/config/pretrain \
  --config-name hubert_base_librispeech \
  task.data=/mnt/lustre/sjtu/home/xc915/superb/dataset/librispeech_finetuning_data/10min task.label_dir=/mnt/lustre/sjtu/home/xc915/superb/wyj-fairseq/examples/hubert/output task.labels='["km"]' model.label_rate=100