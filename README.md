# Cryptocurrency-Price-Prediction
## Description
- Implementation of cryptocurrency price prediction model based on CRNN(Convolutional Recurrent Neural Network).

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
1. Edit config.json like below. Note that you need to set mode to `predict` and set `weight_dir` appropriately. There is a pre-trained best model `../resources/weights_best.pt` for `hidden_dim = 128` and `num_layers = 4`.
```json
{
    "mode" : "predict",
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

## Model Architecture
- 3 conv1d layers -> `num_layers`-layerd LSTM -> 2 fully connected layers with ReLU
- 3 conv1d layers
    - kernel_size = 3, padding = 1 to keep same size
    - in_channels = 1, `hidden_dim`, `hidden_dim`, respectively
    - out_channels = `hidden_dim`, `hidden_dim`, 1, respectively
    - ReLU activation function between all layers
- `num_layers`-layerd LSTM
    - input_size = 1
    - hidden_size = `hiddem_dim`
    - dropout_rate = 0.3
- 2 fully connected layers with ReLU
    - in_features = `hidden_dim`, 32, respectively
    - out_features = 32, 1, respectively
    - ReLU activation function between all layers

## Experimental Results
### Train
<div align="center"><img src="https://github.com/drumpt/Cryptocurrency-Price-Prediction/blob/main/img/training_result.png" width="400"></div>

### Predict
- All models are trained using ethereum price dataset.

| ethereum | ethereum<br />(different look_back) | bitcoin | dogecoin |
| :-------------------------: | :-------------------------:| :-------------------------:| :-------------------------:|
| ![ethereum_to_ethereum](https://github.com/drumpt/Cryptocurrency-Price-Prediction/blob/main/img/ethereum_to_ethereum.png) | ![ethereum_to_ethereum_different_lookback](https://github.com/drumpt/Cryptocurrency-Price-Prediction/blob/main/img/ethereum_to_ethereum_different_lookback.png) | ![ethereum_to_bitcoin](https://github.com/drumpt/Cryptocurrency-Price-Prediction/blob/main/img/ethereum_to_bitcoin.png) | ![ethereum_to_dogecoin](https://github.com/drumpt/Cryptocurrency-Price-Prediction/blob/main/img/ethereum_to_dogecoin.png) |

## Discussion
- I first tried to train the model on the entire dataset, but it was very slow to train and the result was bad. So I trained only on ethereum and predicted the price of other cryptocurrencies.
- The results were better than I thought. Trained model predicted the price of ethereum as well as other cryptocurrencies like bitcoin and dogecoin properly. You can see the result graph above. It also worked well with different look_backs.
- It was possible to predict other cryptocurrencies not only because the model's performance is good, but also the prices of cryptocurrencies tend to follow each other.
- It would be helpful to devise a general-purpose model architecture that can be learned for the entire dataset.

## References
- [Bitcoin Price Prediction](https://wikidocs.net/53275)
- [Stock price prediction using LSTM, RNN and CNN-sliding window model](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8126078)
