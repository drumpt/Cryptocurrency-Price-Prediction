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
    def __init__(self, mode, data_dir, output_dir, weight_dir, epochs):
        self.mode = mode
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.weight_dir = weight_dir
        self.epochs = epochs

        self.model = LSTMPredictor(hidden_dim1 = 128, hidden_dim2 = 128)
        self.model.to(device)

        self.dataset = CryptocurrencyDataset(self.data_dir)
        # self.dataloader = DataLoader(self.dataset, batch_size = 1, shuffle = True)

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(params = self.model.parameters(), lr = 1e-4)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.95)

    def train(self):
        train_size = int(0.8 * len(self.dataset))
        validation_size = len(self.dataset) - train_size
        train_set, validation_set = torch.utils.data.random_split(self.dataset, [train_size, validation_size])
        train_dataloader = DataLoader(train_set, batch_size = 1, shuffle = True)
        validation_dataloader = DataLoader(validation_set, batch_size = 1, shuffle = True)

        history = {"loss" : [], "val_loss" : []}

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            epoch_train_loss = 0
            epoch_validation_loss = 0

            # train
            for item in tqdm(train_dataloader):
                self.model.zero_grad()
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

            self.scheduler.step()
            self.save_model(epoch + 1, epoch_train_loss, epoch_validation_loss)

        self.plot(history)

    def test(self):
        

    def predict(self):
        pass

    def save_model(self, epoch, loss, val_loss):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        output_path = os.path.join(self.output_dir, f"weights_{epoch:03d}_{loss:.4f}_{val_loss:.4f}.pt")
        torch.save(self.model.state_dict(), output_path)
        return output_path

    def plot(self, history):
        plt.title("Loss versus epoch")
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
        ratio = self.hidden2ratio(lstm2_out)
        return ratio

class CryptocurrencyDataset(Dataset):
    def __init__(self, data_dir):
        self.csv_files = preprocessor.get_all_csv_files(data_dir)
    
    def __len__(self):
        return len(self.csv_files)
    
    def __getitem__(self, idx):
        input_tensor, output_tensor = preprocessor.read_csv(self.csv_files[idx])
        item = {"input_tensor" : input_tensor.unsqueeze(1), "output_tensor" : output_tensor.unsqueeze(1)}
        return item