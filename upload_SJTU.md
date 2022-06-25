model_url=https://dl.fbaipublicfiles.com/fairseq/data2vec/audio_base_ls.pt # change your model url
model=${model_url##*/}
wget $model_url tmp/${model}
scp tmp/${model} wangyujin@101.6.68.159:~/models/