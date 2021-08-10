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
- https://wikidocs.net/53275
