import json
import argparse

import model

def main(config):
    mode = config["mode"]
    predictor = model.CryptocurrencyPricePredictor(config)
    if mode == "train":
        predictor.train()
    elif mode == "test":
        predictor.test()
    elif mode == "predict":
        predictor.predict()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type = str, required = False, default = "../config.json")
    args = parser.parse_args()

    with open(args.config_dir) as config_file:
        config = json.load(config_file)
    main(config)