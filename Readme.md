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

# slurm を使ってモデル生成
sbatch tensor.batch

# モデルの生成
make build-model

# 文章の生成
make run 
```
