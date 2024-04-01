import torch
import pickle
from model import GRU, officialGRU
from datasets import elecDataset, multiflowDataset, weatherDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


class trainApp():

    def __init__(self, use_cuda=True, epochs=16, seq_length=32, batch_size=16, state_pth=None, csv_dir=None):
        self.realList = ["elec", "weather"]
        self.use_cuda = use_cuda
        self.seq_length = seq_length
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.train_dataset, self.val_dataset = self.initDataset(csv_dir)
        self.feature_dim = self.train_dataset.feature_dim
        self.class_dim = self.train_dataset.class_dim

        if state_pth is None:
            self.model = self.initModel(feature_dim=self.feature_dim, class_dim=self.class_dim)
        else:
            self.model = self.loadModel(feature_dim=self.feature_dim, class_dim=self.class_dim, state_pth=state_pth)

        self.optimizer = self.initOptimizer()
        self.scheduler = self.initScheduler()
        self.epochs = epochs
        self.batch_size = batch_size

    def initDataset(self, csv_dir):
        if csv_dir not in self.realList:
            train_dataset = multiflowDataset(csv_dir=csv_dir, seq_length=self.seq_length, train_ratio=0.8, online=False,
                                             train=True)
            val_dataset = multiflowDataset(csv_dir=csv_dir, seq_length=self.seq_length, train_ratio=0.8, online=False,
                                           train=False)
        else:
            if csv_dir == "elec":
                train_dataset = elecDataset(seq_length=self.seq_length, train_ratio=0.8, online=False, train=True)
                val_dataset = elecDataset(seq_length=self.seq_length, train_ratio=0.2, online=False, train=False)
            else:
                train_dataset = weatherDataset(seq_length=self.seq_length, train_ratio=0.8, online=False, train=True)
                val_dataset = weatherDataset(seq_length=self.seq_length, train_ratio=0.2, online=False, train=False)
        return train_dataset, val_dataset

    def loadModel(self, feature_dim, class_dim, state_pth):
        model = officialGRU(input_dim=feature_dim, output_dim=class_dim, memory_size=128)
        model.load_state_dict(state_dict=torch.load(state_pth))
        model = model.to(self.device)
        return model

    def initModel(self, feature_dim, class_dim):
        model = officialGRU(input_dim=feature_dim, output_dim=class_dim, memory_size=128)
        model = model.to(self.device)
        return model

    def initTrainDl(self):
        dataset = self.train_dataset
        train_dl = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        return train_dl

    def initValDl(self):
        dataset = self.val_dataset
        val_dl = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        return val_dl

    def initOptimizer(self):
        optimizer = Adam(params=self.model.parameters(), lr=0.001, weight_decay=0.001)
        return optimizer

    def initScheduler(self):
        scheduler = StepLR(self.optimizer, step_size=2, gamma=0.1)
        return scheduler

    def doTraining(self):
        self.model.train()
        loss_func = CrossEntropyLoss()
        loss_record = []
        train_dl = self.initTrainDl()
        for epoch in tqdm(range(self.epochs)):
            for data in train_dl:
                self.optimizer.zero_grad()
                x, y = data
                x_g = x.to(self.device)
                y_g = y.to(self.device)
                logit, probability = self.model(x_g)
                loss = loss_func(logit, y_g)
                loss.backward()
                self.optimizer.step()
            loss_record.append(loss.detach().to("cpu").item())
            print(loss.item())

        return loss_record

    def doValidation(self):
        self.model.eval()
        loss_func = CrossEntropyLoss()
        val_dl = self.initValDl()
        sample_count = 0
        true_count = 0
        acc_record = []
        # pred_result = torch.zeros([sample_count])
        with torch.no_grad():
            for data in tqdm(val_dl):
                x, y = data
                sample_count += x.size(0)
                x_g = x.to(self.device)
                y_g = y.to(self.device)
                logit, probability = self.model(x_g)
                loss = loss_func(logit, y_g)
                true_pred_mask = (torch.argmax(probability, dim=1) == torch.argmax(y_g, dim=1))
                true_count += torch.sum(true_pred_mask).item()
                acc = true_count / sample_count
                acc_record.append(acc)
        return acc_record


if __name__ == "__main__":
    app = trainApp(csv_dir="weather")
    loss_record = app.doTraining()
    print("Validation")
    acc_record = app.doValidation()
    with open("weather_pretrained_record.pkl", "wb") as f:
        pickle.dump(acc_record, f)
    app.model.to(torch.device('cpu'))
    torch.save(app.model.state_dict(), "weather_pretrained_GRU.pth")
    print(acc_record)