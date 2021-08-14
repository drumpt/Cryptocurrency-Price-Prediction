# Cryptocurrency-Price-Prediction
## Description
Implementation of cryptocurrency price prediction model based on CRNN(Convolutional Recurrent Neural Network).

## Requirements
```txt
python>=3.7
torch
sklearn
matplotlib
tqdm
```

## Usage
### Train
1. Edit config.json like below. Note that you need to set mode to `train`.
```json
{
    "mode" : "train",
    "data_dir" : "../datasets",
    "output_dir" : "../resources",
    "weight_dir" : "",
    "batch_size" : 4,
    "epochs" : 30,
    "look_back" : 30,
    "hidden_dim" : 128,
    "num_layers" : 4
}
```

2. Type the command below.
```
cd path_to_this_repository/src
python3 main.py
```


### Test
1. Edit config.json like below. Note that you need to set mode to `test` and set `weight_dir` appropriately. There is a pre-trained best model `../resources/weights_best.pt` for `hidden_dim = 128` and `num_layers = 4`.
```json
{
    "mode" : "test",
    "data_dir" : "../datasets",
    "output_dir" : "../resources",
    "weight_dir" : "WEIGHT_DIR",
    "batch_size" : 4,
    "epochs" : 30,
    "look_back" : 30,
    "hidden_dim" : 128,
    "num_layers" : 4
}
```

2. Type the command below.
```
cd path_to_this_repository/src
python3 main.py
```

### Predict
1. Edit config.json like below. Note that you need to set mode to `test` and set `weight_dir` appropriately. 1. There is a pre-trained best model `../resources/weights_best.pt` for `hidden_dim = 128` and `num_layers = 4`.
```json
{
    "mode" : "test",
    "data_dir" : "../datasets",
    "output_dir" : "../resources",
    "weight_dir" : "WEIGHT_DIR",
    "batch_size" : 4,
    "epochs" : 30,
    "look_back" : 30,
    "hidden_dim" : 128,
    "num_layers" : 4
}
```

2. Type the command below.
```
cd path_to_this_repository/src
python3 main.py
```

## Experimental Results
### Train


### Predict



## References
- [비트코인 가격 예측](https://wikidocs.net/53275)
- [Stock price prediction using LSTM, RNN and CNN-sliding window model](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8126078)
