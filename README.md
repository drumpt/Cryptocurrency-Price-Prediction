# Cryptocurrency-Price-Prediction
## Description
This project is a toy project for understanding and implementing RNN models.

## Usage
### Train
```
python3 main.py --mode train --data_dir ../datasets --output_dir ../resources --batch_size 4 --epochs 30 --look_back 30
```

### Test
```
python3 main.py --mode test --data_dir ../datasets --weight_dir WEIGHT_DIR --batch_size 4 --look_back 30
```

### Predict
```
python3 main.py --mode predict --data_dir ../datasets/coin_Bitcoin.csv --weight_dir WEIGHT_DIR --look_back 30
```

## Result

## References
- [비트코인 가격 예측](https://wikidocs.net/53275)
- [Stock price prediction using LSTM, RNN and CNN-sliding window model](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8126078)
