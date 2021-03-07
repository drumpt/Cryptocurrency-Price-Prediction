import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

import preprocessor

device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cpu":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

class CryptocurrencyPricePredictor():
    def __init__(self, mode, data_dir, output_dir, weight_dir, batch_size, epochs, look_back):
        self.mode = mode
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.weight_dir = weight_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.look_back = look_back

        self.model = LSTMPredictor(hidden_dim1 = 128, hidden_dim2 = 128)
        self.model.to(device)
        print(self.model)

        if not self.mode == "train" and os.path.exists(self.weight_dir):
            self.model.load_state_dict(torch.load(self.weight_dir))

        if self.mode in ["train", "test"]:
            self.dataset = CryptocurrencyDataset(self.data_dir)

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(params = self.model.parameters(), lr = 1e-5)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.95)

    def train(self):
        train_size = int(0.8 * len(self.dataset))
        validation_size = len(self.dataset) - train_size
        train_set, validation_set = torch.utils.data.random_split(self.dataset, [train_size, validation_size])
        train_dataloader = DataLoader(train_set, batch_size = self.batch_size, shuffle = True)
        validation_dataloader = DataLoader(validation_set, batch_size = self.batch_size, shuffle = True)

        history = {"loss" : [], "val_loss" : []}

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            epoch_train_loss = 0
            epoch_validation_loss = 0

            # train
            for item in tqdm(train_dataloader):
                self.optimizer.zero_grad()

                input_tensor, target_tensor = item["input_tensor"], item["output_tensor"]
                output_tensor = self.model(input_tensor)

                loss = self.loss_function(output_tensor, target_tensor)
                loss.backward()
                epoch_train_loss += loss
                self.optimizer.step()

            # validation
            for item in tqdm(validation_dataloader):
                input_tensor, target_tensor = item["input_tensor"], item["output_tensor"]
                output_tensor = self.model(input_tensor)

                loss = self.loss_function(output_tensor, target_tensor)
                epoch_validation_loss += loss

            epoch_train_loss /= len(train_dataloader)
            epoch_validation_loss /= len(validation_dataloader)

            history["loss"].append(epoch_train_loss)
            history["val_loss"].append(epoch_validation_loss)

            print(f"loss : {epoch_train_loss:.4f}, val_loss : {epoch_validation_loss:.4f}")
            self.save_model(epoch + 1, epoch_train_loss, epoch_validation_loss)

        self.plot(history)

    def test(self):
        dataloader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True)
        test_loss = 0

        for item in tqdm(dataloader):
            input_tensor, target_tensor = item["input_tensor"], item["output_tensor"]
            output_tensor = self.model(input_tensor)

            test_loss += self.loss_function(output_tensor, target_tensor)

        test_loss /= len(dataloader)
        print(f"test_loss : {test_loss:.4f}")

    def predict(self):
        input_ratio_list, target_ratio_list = preprocessor.read_csv(self.data_dir, sort = "ratio", look_back = self.look_back)
        input_price_list, target_price_list = preprocessor.read_csv(self.data_dir, sort = "price", look_back = self.look_back)
        output_ratio_list, output_price_list = [], []

        for i in range(len(input_ratio_list)):
            output_ratio = self.model(torch.tensor(input_ratio_list[i], dtype = torch.float).unsqueeze(0).unsqueeze(-1)).detach()
            output_ratio_list.append(output_ratio.squeeze(-1).squeeze(0))

        print(output_ratio_list)

        output_price_list = target_price_list[0 : self.look_back]
        for i in range(len(output_ratio_list)):
            output_price = (output_ratio_list[i] + 1) * output_price_list[-1]
            output_price_list.append(output_price)

        plt.title("predicted price vs real price")
        plt.plot(target_price_list)
        plt.plot(output_price_list)
        plt.legend(['real price', 'predicted price'])
        plt.xlabel('day')
        plt.ylabel('price')
        plt.show()
        plt.savefig(os.path.join(self.output_dir, "predict_result.png"))

    def save_model(self, epoch, loss, val_loss):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        output_path = os.path.join(self.output_dir, f"weights_{epoch:03d}_{loss:.4f}_{val_loss:.4f}.pt")
        torch.save(self.model.state_dict(), output_path)
        return output_path

    def plot(self, history):
        plt.title("Loss vs epoch")
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.legend(['loss', 'val_loss'], loc = 'upper right')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(self.output_dir, "training_result.png"))

class LSTMPredictor(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2):
        super(LSTMPredictor, self).__init__()

        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        self.lstm1 = nn.LSTM(1, hidden_dim1)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2)
        self.hidden2ratio = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        lstm1_out, _ = self.lstm1(x)
        lstm2_out, _ = self.lstm2(lstm1_out)
        ratio = self.hidden2ratio(lstm2_out[:, -1, :])
        return ratio

class CryptocurrencyDataset(Dataset):
    def __init__(self, data_dir):
        self.csv_files = preprocessor.get_all_csv_files(data_dir)
        self.input_list, self.output_list = [], []
        for csv_file in self.csv_files:
            input_list, output_list = preprocessor.read_csv(csv_file, sort = "ratio", look_back = self.look_back)
            self.input_list.extend(input_list)
            self.output_list.extend(output_list)
        self.input_list = torch.tensor(self.input_list, dtype = torch.float).unsqueeze(-1)
        self.output_list = torch.tensor(self.output_list, dtype = torch.float).unsqueeze(-1)

    def __len__(self):
        return self.input_list.shape[0]

    def __getitem__(self, idx):
        item = {
            "input_tensor" : self.input_list[idx],
            "output_tensor" : self.output_list[idx]
        }
        return item