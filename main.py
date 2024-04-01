import torch
import pickle
from util import OnlineClassificationEvaluation, multiflowOnlineClassificationEvaluation
from RealDriftGenerator import RealDriftGenerator
from skmultiflow.bayes import NaiveBayes
from skmultiflow.bayes import NaiveBayes
from datasets import elecDataset
from model import officialGRU
import pandas as pd


def evaluateRNN():
    model = officialGRU(input_dim=5, output_dim=2, memory_size=128)

    model.load_state_dict(state_dict=torch.load("./pretrained_GRU.pth"))

    drift_dict = {700: (100, "middle")}
    reverse_elec = elecDataset(online=True, drift_dict=drift_dict, stream_length=1000)
    loss_record, acc_record = OnlineClassificationEvaluation(model, reverse_elec)
    with open("online_record_reverse_smooth.pkl", "wb") as f:
        pickle.dump(acc_record, f)


def evaluateClassifier():
    elec_df = pd.read_csv("./electricity-normalized.csv", nrows=2000,
                          usecols=["nswprice", "nswdemand", "vicprice", "vicdemand", "transfer", "class"])
    elec_df['class'] = elec_df['class'].apply(lambda element: 1 if element == "UP" else 0)
    elec_generator = RealDriftGenerator(df=elec_df)
    drift_dict = {500: (100, "middle")}
    reversed_df = elec_generator.reverseSlice(drift_dict=drift_dict)
    bayes_acc_record = multiflowOnlineClassificationEvaluation(df=reversed_df, classifier=NaiveBayes())


evaluateRNN()
