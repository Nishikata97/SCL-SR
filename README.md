# DIREC

## Code

This is the source code for the Paper: _Semantic Relation Guided Dual-view Contrastive Learning for Session-based Recommendations_.

## Requirements

- Python 3.9
- PyTorch 1.12

## Best Hyperparameter:
- Tmall: alpha=0.02, beta=100, num_coarse_sampling=90
- E-Commerce (CIKM19): alpha=0.02, beta=150, num_coarse_sampling=90
- Cosmetics: alpha=0.02, beta=100, num_coarse_sampling=90

## Datasets:
| Dataset    | URLs                                                         |
| :--------- | :----------------------------------------------------------- |
| Tmall      | https://tianchi.aliyun.com/dataset/dataDetail?dataId=42      |
| Cosmetics  | https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop |
| E-Commerce | https://tianchi.aliyun.com/competition/entrance/231721/information |



## Train & Test:
- For Tmall and E-Commerce(CIKM19):
~~~~
python main.py --dataset Tmall --alpha 0.02 --beta 100
python main.py --dataset CIKM19 --alpha 0.02 --beta 150
~~~~
- For Cosmetics:
~~~~
python main_cosmetics.py --dataset Cosmetics --alpha 0.02 --beta 100
~~~~

## Preprocessed data:
The datasets have been preprocessed and encoded with pickle, which can be downloaded from the [link](https://pan.baidu.com/s/1eJXDmYdPiyWgEDFcoEFstg) (password: v3bj)
