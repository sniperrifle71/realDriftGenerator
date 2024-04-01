import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from util import reverseSlice
from RealDriftGenerator import RealDriftGenerator




class multiflowDataset(Dataset):
    def __init__(self, csv_dir, seq_length=8, train=True, online=True, train_ratio=0.8):
        super(multiflowDataset, self).__init__()
        self.seq_length = seq_length
        self.csv_dir = csv_dir
        self.tupleList, class_dim, self.feature_dim = self.initTupleList(self.csv_dir)
        self.class_dim = int(class_dim)
        self.seqList = self.divideSequence()
        if not online:
            if train:
                self.seqList = self.seqList[0: int(len(self.seqList) * train_ratio)]
            else:
                self.seqList = self.seqList[0: int(len(self.seqList) * (1 - train_ratio))]

    def initTupleList(self, csv_dir):
        tupleList = []
        df = pd.read_csv(csv_dir)
        features_mean = df.iloc[:, :-1].mean()
        features_std = df.iloc[:, :-1].std()
        df.iloc[:, :-1] = (df.iloc[:, :-1] - features_mean) / features_std

        class_dim = df.iloc[:, df.shape[1] - 1].max()+1
        feature_dim = df.shape[1] - 1
        for i in range(df.shape[0]):
            features = df.iloc[i, 0:df.shape[1] - 1]
            label = df.iloc[i, df.shape[1] - 1]
            tupleList.append((features, label))
        return tupleList, class_dim, feature_dim

    def divideSequence(self):
        divide_idx = 0
        seq_list = []
        while divide_idx + self.seq_length < len(self.tupleList):
            seq = torch.zeros([self.seq_length, self.feature_dim+1])
            seqTuple = self.tupleList[divide_idx: divide_idx + self.seq_length]
            for i in range(self.seq_length):
                seq[i, :self.feature_dim] = torch.from_numpy(seqTuple[i][0].values.astype(np.float32))
                seq[i, self.feature_dim] = torch.from_numpy(np.array(seqTuple[i][1], dtype=np.float32))
            divide_idx += 1
            seq_list.append(seq)
        return seq_list

    def __len__(self):
        return len(self.seqList)

    def __getitem__(self, item):
        features = self.seqList[item][:, :self.feature_dim]
        label = torch.eye(int(self.class_dim))[int(self.seqList[item][-1, self.feature_dim])]
        return features, label


class elecDataset(Dataset):
    def __init__(self, seq_length=8, train=True, online=False, train_ratio=0.8, drift_dict=None, stream_length=1000):
        super(elecDataset, self).__init__()
        self.seq_length = seq_length
        self.stream_length = stream_length
        self.tupleList, self.feature_dim, self.class_dim = self.initTupleList(drift_dict)
        self.seqList = self.divideSequence()
        if not online:
            if train:
                self.seqList = self.seqList[0: int(len(self.seqList) * train_ratio)]
            else:
                self.seqList = self.seqList[0: int(len(self.seqList) * (1 - train_ratio))]

    def initTupleList(self, drift_dict):
        tupleList = []
        df = pd.read_csv("./electricity-normalized.csv",
                         usecols=["nswprice", "nswdemand", "vicprice", "vicdemand", "transfer", "class"],
                         nrows=self.stream_length)

        df["class"] = df["class"].apply(lambda element: 1 if element == "UP" else 0)
        feature_dim = df.shape[1]-1
        class_dim = int(df.iloc[:, feature_dim].max())+1
        stream = RealDriftGenerator(df)

        if drift_dict is not None:
            df = stream.reverseSlice(drift_dict=drift_dict)
        else:
            df = stream.origin_df

        column = df.shape[1]
        for i in range(df.shape[0]):
            features = df.iloc[i, 0:column - 1]
            label = df.iloc[i, column - 1]
            tupleList.append((features, label))
        return tupleList, feature_dim, class_dim

    def divideSequence(self):
        divide_idx = 0
        seq_list = []
        while divide_idx + self.seq_length < len(self.tupleList):
            seq = torch.zeros([self.seq_length, 6])
            seqTuple = self.tupleList[divide_idx: divide_idx + self.seq_length]
            for i in range(self.seq_length):
                seq[i, :5] = torch.from_numpy(seqTuple[i][0].values.astype(np.float32))
                seq[i, 5] = torch.from_numpy(np.array(seqTuple[i][1], dtype=np.float32))
            divide_idx += 1
            seq_list.append(seq)
        return seq_list

    def __len__(self):
        return len(self.seqList)

    def __getitem__(self, item):
        features = self.seqList[item][:, :5]
        label = torch.eye(2)[int(self.seqList[item][-1, 5])]
        return features, label

class weatherDataset(Dataset):
    def __init__(self, seq_length=8, train=True, online=False, train_ratio=0.8, drift_dict=None, stream_length=1000):
        super(weatherDataset, self).__init__()
        self.seq_length = seq_length
        self.stream_length = stream_length
        self.tupleList, self.feature_dim, self.class_dim = self.initTupleList(drift_dict)
        self.seqList = self.divideSequence()
        if not online:
            if train:
                self.seqList = self.seqList[0: int(len(self.seqList) * train_ratio)]
            else:
                self.seqList = self.seqList[0: int(len(self.seqList) * (1 - train_ratio))]

    def initTupleList(self, drift_dict):
        tupleList = []

        df = pd.read_csv("./seattle-weather.csv", nrows=self.stream_length,
                                 usecols=["precipitation", "temp_max", "temp_min", "wind", "weather"])
        class_dict = {"drizzle": 0, "rain": 1, "sun": 2, "snow": 3, "fog": 4}
        df["weather"] = df["weather"].apply(lambda x: class_dict[x])

        feature_dim = df.shape[1]-1
        class_dim = int(df.iloc[:, feature_dim].max())+1
        stream = RealDriftGenerator(df)

        if drift_dict is not None:
            df = stream.reverseSlice(drift_dict=drift_dict)
        else:
            df = stream.origin_df

        column = df.shape[1]
        for i in range(df.shape[0]):
            features = df.iloc[i, 0:column - 1]
            label = df.iloc[i, column - 1]
            tupleList.append((features, label))
        return tupleList, feature_dim, class_dim

    def divideSequence(self):
        divide_idx = 0
        seq_list = []
        while divide_idx + self.seq_length < len(self.tupleList):
            seq = torch.zeros([self.seq_length, 5])
            seqTuple = self.tupleList[divide_idx: divide_idx + self.seq_length]
            for i in range(self.seq_length):
                seq[i, :4] = torch.from_numpy(seqTuple[i][0].values.astype(np.float32))
                seq[i, 4] = torch.from_numpy(np.array(seqTuple[i][1], dtype=np.float32))
            divide_idx += 1
            seq_list.append(seq)
        return seq_list

    def __len__(self):
        return len(self.seqList)

    def __getitem__(self, item):
        features = self.seqList[item][:, :4]
        label = torch.eye(5)[int(self.seqList[item][-1, 4])]
        return features, label


if __name__ == "__main__":
    drift_dict = {500: (100, "middle"), 1000: (100, "left"), 1500: (200, "right")}
    dset = elecDataset(drift_dict=drift_dict, online=True)
    #csv_dir = "./AGRAWAL_p700_w100_l1000.csv"
    #df = pd.read_csv(csv_dir)
    #waveform_dset = multiflowDataset(csv_dir=csv_dir)
    print(dset.feature_dim, dset.class_dim)
