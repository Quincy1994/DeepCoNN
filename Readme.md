# DeepCoNN

This is the implement for DeepCoNN, originated from https://github.com/chechongthu/DeepCoNN

## Environments

- python >= 3.5.0
- Tensorflow >= 1.4.0
- numpy
- pandas
- gensim

## 1. Data Preprocessing
Preprocess data for training model
### 1.1 split data as train set, valid set and test set;
``` sh
python3 loaddata.py
```

### 1.2 get pre-trained word embedding
``` sh
python3 vocab_and_emb.py
```

### 1.3 get hyperparameters of dataset
``` sh
python3 data_pro.py
```

## 2.Training
train model
``` sh
python3 train.py
```


### Main directories and files

- `./data`: storage for datasets and embeddings
- './src/data_preprocess': codes for preprocessing data
- './src/DeeCoNN': codes for building model and training
