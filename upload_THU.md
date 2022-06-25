1. Download model to WSL

```bash
$ wget https://dl.fbaipublicfiles.com/fairseq/data2vec/audio_base_ls.pt
```

2. Upload model to jump machine

```bash
$ scp <model_name> wangyujin@101.6.68.159:.
```

3. Upload model to target machine

```bash
$ scp <model_name> g2:~/models/
```