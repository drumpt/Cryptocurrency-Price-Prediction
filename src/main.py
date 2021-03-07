import argparse

import model

def main(mode, data_dir, output_dir, weight_dir, epochs):
    predictor = model.CryptocurrencyPricePredictor(mode, data_dir, output_dir, weight_dir, epochs)
    if mode == "train":
        predictor.train()
    elif mode == "test":
        predictor.test()
    else: # predict
        predictor.predict()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type = str, required = True)
    parser.add_argument("--data_dir", type = str, required = False, default = "../datasets")
    parser.add_argument("--output_dir", type = str, required = False, default = "../resources")
    parser.add_argument("--weight_dir", type = str, required = False, default = "../resources/weights_best.pt")
    parser.add_argument("--epochs", type = int, required = False, default = 30)
    args = parser.parse_args()

    mode = args.mode
    data_dir = args.data_dir
    output_dir = args.output_dir
    weight_dir = args.weight_dir
    epochs = args.epochs

    main(mode, data_dir, output_dir, weight_dir, epochs)