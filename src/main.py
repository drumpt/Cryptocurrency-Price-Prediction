import argparse

import model

def main(mode, data_dir, output_dir, weight_dir, batch_size, epochs, look_back):
    predictor = model.CryptocurrencyPricePredictor(mode, data_dir, output_dir, weight_dir, batch_size, epochs, look_back)
    if mode == "train":
        predictor.train()
    elif mode == "test":
        predictor.test()
    elif mode == "predict":
        predictor.predict()
    else: # debug
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type = str, required = True)
    parser.add_argument("--data_dir", type = str, required = False, default = "../datasets")
    parser.add_argument("--output_dir", type = str, required = False, default = "../resources")
    parser.add_argument("--weight_dir", type = str, required = False, default = "")
    parser.add_argument("--batch_size", type = int, required = False, default = 8)
    parser.add_argument("--epochs", type = int, required = False, default = 30)
    parser.add_argument("--look_back", type = int, required = False, default = 30)
    args = parser.parse_args()

    mode = args.mode
    data_dir = args.data_dir
    output_dir = args.output_dir
    weight_dir = args.weight_dir
    batch_size = args.batch_size
    epochs = args.epochs
    look_back = args.look_back

    main(mode, data_dir, output_dir, weight_dir, batch_size, epochs, look_back)