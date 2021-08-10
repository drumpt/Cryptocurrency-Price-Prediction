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
print(f"device : {device}")

class CryptocurrencyPricePredictor():
    def __init__(self, config):
        self.mode = config["mode"]
        self.data_dir = config["data_dir"]
        self.output_dir = config["output_dir"]
        self.weight_dir = config["weight_dir"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.look_back = config["look_back"]

        self.model = LSTMPredictor(config)
        self.model.to(device)
        print(self.model)

        if not self.mode == "train" and os.path.exists(self.weight_dir):
            self.model.load_state_dict(torch.load(self.weight_dir))

        if self.mode in ["train", "test"]:
            self.dataset = CryptocurrencyDataset(self.data_dir, self.look_back)

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(params = self.model.parameters(), lr = 1e-4)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.95)

    def train(self):
        train_size = int(0.8 * len(self.dataset))
        validation_size = len(self.dataset) - train_size
        train_set, validation_set = torch.utils.data.random_split(self.dataset, [train_size, validation_size], generator = torch.Generator(device = "cuda:0").manual_seed(42))
        train_dataloader = DataLoader(train_set, batch_size = self.batch_size, shuffle = False)
        validation_dataloader = DataLoader(validation_set, batch_size = self.batch_size, shuffle = False)

        history = {"loss" : [], "val_loss" : []}

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            epoch_train_loss = 0
            epoch_validation_loss = 0

            # train
            for item in tqdm(train_dataloader):
                input_tensor, target_tensor = item["input_tensor"], item["output_tensor"]
                output_tensor = self.model(input_tensor)

                loss = self.loss_function(output_tensor, target_tensor)

                self.optimizer.zero_grad()
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

            print(f"loss : {epoch_train_loss:.6f}, val_loss : {epoch_validation_loss:.6f}")
            self.save_model(epoch + 1, epoch_train_loss, epoch_validation_loss)

        self.plot(history)

    def test(self):
        dataloader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle = False)
        test_loss = 0

        for item in tqdm(dataloader):
            input_tensor, target_tensor = item["input_tensor"], item["output_tensor"]
            output_tensor = self.model(input_tensor)

            test_loss += self.loss_function(output_tensor, target_tensor)

        test_loss /= len(dataloader)
        print(f"test_loss : {test_loss:.6f}")

    def predict(self):
        input_ratio_list, target_ratio_list = preprocessor.read_csv(self.data_dir, look_back = self.look_back, sort = "ratio")
        input_price_list, target_price_list = preprocessor.read_csv(self.data_dir, look_back = self.look_back, sort = "price")
        output_ratio_list, output_price_list = [], []

        for i in range(len(input_ratio_list)):
            output_ratio = self.model(torch.tensor(input_ratio_list[i], dtype = torch.float).unsqueeze(0).unsqueeze(2)).detach()
            output_ratio_list.append(output_ratio.squeeze(0).squeeze(0))

        output_price_list = target_price_list[0 : self.look_back]
        for i in range(len(output_ratio_list)):
            output_price = (output_ratio_list[i] + 1) * output_price_list[-1]
            output_price_list.append(output_price)

        plt.title("predicted price vs real price")
        # plt.plot(target_ratio_list)
        # plt.plot(output_ratio_list)
        plt.plot(target_price_list)
        plt.plot(output_price_list)
        plt.legend(['real price', 'predicted price'])
        plt.xlabel('day')
        plt.ylabel('price')
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

# class LSTMPredictor(nn.Module):
#     def __init__(self, hidden_dim, num_layers, look_back):
#         super(LSTMPredictor, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers

#         self.conv1 = nn.Conv1d(in_channels = 1, out_channels = hidden_dim, kernel_size = 3, padding = 1)
#         self.conv2 = nn.Conv1d(in_channels = hidden_dim, out_channels = hidden_dim, kernel_size = 3, padding = 1)
#         self.conv3 = nn.Conv1d(in_channels = hidden_dim, out_channels = 1, kernel_size = 3, padding = 1)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)
#         self.lstm = nn.LSTM(look_back, hidden_dim, num_layers, batch_first = True)
#         self.fc1 = nn.Linear(hidden_dim, 32)
#         self.fc2 = nn.Linear(32, 1)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         # x = x.reshape((x.shape[0], x.shape[2], x.shape[1]))
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.relu(self.conv3(x))

#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, requires_grad = True).detach()
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, requires_grad = True).detach()
#         out, (hn, cn) = self.lstm(x, (h0, c0))
#         out = self.relu(self.fc1(out[:, -1, :]))
#         out = self.fc2(out)
#         return out

class LSTMPredictor(nn.Module):
    def __init__(self, config):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config["num_layers"]
        self.batch_size = config["batch_size"]
        self.look_back = config["look_back"]

        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = self.hidden_dim, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv1d(in_channels = self.hidden_dim, out_channels = self.hidden_dim, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv1d(in_channels = self.hidden_dim, out_channels = 1, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.lstm = nn.LSTM(self.look_back, self.hidden_dim, self.num_layers, dropout = 0.3)
        self.fc1 = nn.Linear(self.hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

        self.h0 = torch.randn(self.num_layers, self.batch_size, self.hidden_dim, requires_grad = True).detach()
        self.c0 = torch.randn(self.num_layers, self.batch_size, self.hidden_dim, requires_grad = True).detach()

    def forward(self, x):
        print(f"first : {x.shape}")
        x = self.relu(self.bn1(self.conv1(x)))
        print(f"second : {x.shape}")
        x = self.relu(self.bn2(self.conv2(x)))
        print(f"third : {x.shape}")
        x = self.relu(self.conv3(x))
        print(f"fourth : {x.shape}")

        x.permute(2, 0, 1) # (seq_len, batch_size, input_size)
        print(f"fifth : {x.shape}")

        out, (hn, cn) = self.lstm(x, (self.h0, self.c0))

        print(f"sixth : {x.shape}")

        out = self.relu(self.fc1(out[:, -1, :]))
        out = self.fc2(out)
        return out

class CryptocurrencyDataset(Dataset):
    def __init__(self, data_dir, look_back):
        self.csv_files = preprocessor.get_all_csv_files(data_dir)
        self.look_back = look_back
        self.input_list, self.output_list = [], []

        for csv_file in self.csv_files:
            input_list, output_list = preprocessor.read_csv(csv_file, look_back = self.look_back, sort = "ratio")
            self.input_list.extend(input_list)
            self.output_list.extend(output_list)

        # self.input_list = torch.tensor(self.input_list, dtype = torch.float).unsqueeze(-1).to(device)
        # self.output_list = torch.tensor(self.output_list, dtype = torch.float).unsqueeze(-1).to(device)
        self.input_list = torch.tensor(self.input_list, dtype = torch.float).unsqueeze(1).to(device)
        self.output_list = torch.tensor(self.output_list, dtype = torch.float).unsqueeze(1).to(device)

        # print(f"self.input_list.shape : {self.input_list.shape}")

    def __len__(self):
        return self.input_list.shape[0]

    def __getitem__(self, idx):
        item = {
            "input_tensor" : self.input_list[idx],
            "output_tensor" : self.output_list[idx]
        }
        return item