## workflow

1. 在本地wsl上修改代码，每到一个milestone就commit一次。
2. 分别向origin(SJTU Server)和server(THU Server)push。
3. 在THU Server和SJTU Server上运行时，先向远程仓库拉取。 

## docker

进入调试界面：

```bash
$ docker run -it --rm --gpus all --name=fairseq_decode -v /mnt:/mnt chenxie95/speechimage:flashlight-v2 /bin/bash
```

直接使用docker运行某脚本：
```bash
$ docker run -it --rm --gpus all --name=fairseq_decode -v /mnt:/mnt chenxie95/speechimage:flashlight-v3 /bin/bash -c 'sh /mnt/lustre/sjtu/home/xc915/superb/wyj-fairseq/audio_ssl.sh'
```
