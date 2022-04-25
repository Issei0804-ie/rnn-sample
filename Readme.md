```
python --version 
Python 3.8.2
```


[こちら](https://www.kaggle.com/datasets/tanreinama/japanese-fakenews-dataset)からデータセットをダウンロードし， この Readme.md と同じ階層( `./fakenews.csv` ) となるように配置してください．

## 実行方法

```
# amane などに ssh している状態
make build-sif
make pip-install

# slum 使うなら
sbatch tensor.batch

# singularity で実行するなら
make run
```
