# random projection

## pretrain

```bash
$ python fairseq_cli/hydra_train.py -m --config-dir examples/random_projection/config/audio/pretraining \
--config-name pretrain task.data=/path/to/manifests common.user_dir=examples/random_projection
```

You can directly run random_projection.sh in root dir after change the path.
