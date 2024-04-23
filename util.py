from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


def OnlineClassificationEvaluation(model, dataset):
    dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=1)
    optimizer = SGD(params=model.parameters(), lr=0.01)
    loss_func = CrossEntropyLoss()
    true_count = 0
    sample_count = 0
    loss_record = []
    acc_record = []
    for data in tqdm(dataloader):
        sample_count += 1
        x, y = data
        model.eval()
        y_pred, logit = model(x)
        model.train()
        optimizer.zero_grad()
        loss = loss_func(logit, y)
        loss.backward()
        optimizer.step()
        if torch.argmax(y_pred, dim=1) == torch.argmax(y, dim=1):
            true_count += 1
        accuracy = true_count / sample_count
        loss_record.append(loss.detach().cpu().item())
        acc_record.append(accuracy)

    return loss_record, acc_record


def standardized_euclidean_distance(p, q):
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    # 计算均值

    mean_p = np.mean(p, axis=0)
    mean_q = np.mean(q, axis=0)

    # 计算标准差
    std_p = np.std(p, axis=0)
    std_q = np.std(q, axis=0)

    standarized_p = (p - mean_p) / std_p
    standarized_q = (q - mean_q) / std_q

    # 计算标准化欧氏距离
    distance = np.sqrt(np.sum(((standarized_p - standarized_q) ** 2), axis=0))
    return distance


def cosine_similarity(p, q):
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    feature_similarities = np.zeros(p.shape[1])
    for feature_idx in range(0, p.shape[1]):
        p_feature = p[:, feature_idx]
        q_feature = q[:, feature_idx]
        distance_p = np.linalg.norm(p_feature)
        distance_q = np.linalg.norm(q_feature)
        dot_product = np.dot(p_feature, q_feature)
        feature_similarity = dot_product / (distance_p * distance_q)
        feature_similarities[feature_idx] = feature_similarity
    return feature_similarities


def multiflowOnlineClassificationEvaluation(df, classifier):
    features_num = len(df.columns) - 1
    classes = [i for i in range(0, int(max(df.iloc[:, features_num])) + 1)]
    features = df.iloc[:, :features_num]
    labels = np.array(df.iloc[:, features_num])
    classifier.reset()
    tru_count = 0
    samples = 0
    acc_record = []
    # random_array = np.random.rand(1, features_num)
    # classifier.partial_fit(random_array, [0], classes=classes)
    for feature, label in zip(features.values, labels):
        samples += 1
        y_pred = classifier.predict([feature])
        classifier.partial_fit([feature], [label], classes=classes)
        if y_pred[0] == label:
            tru_count += 1
        acc = tru_count / samples
        acc_record.append(acc)

    return acc_record


def reverseSlice(df, slice_idx, smooth = False):
    reverse_df = pd.DataFrame(index=range(df.shape[0]), columns=df.columns, dtype=np.float32)
    drift_width = 100
    slice_idx.append(0)
    slice_idx.append(df.shape[0])
    slice_idx.sort(reverse=True)
    head_pointer = 0

    for i in range(0, len(slice_idx) - 1):
        period_length = slice_idx[i] - slice_idx[i + 1]
        reverse_df.iloc[head_pointer: head_pointer + period_length, :] = df.iloc[slice_idx[i + 1]:slice_idx[i], :]
        head_pointer = head_pointer + period_length

        if i != 0 and i != df.shape[0] and smooth:
            reverse_df.iloc[slice_idx[i] - drift_width // 2: slice_idx[i] + drift_width // 2, :].apply(midDriftSmooth,
                                                                                                       axis=0)
    return reverse_df.iloc[::-1].reset_index(drop=True)


def midDriftSmooth(drift_area):
    smooth_drift = drift_area
    right_ewm = drift_area.ewm(span=5).mean()
    right_fit_values = right_ewm.iloc[len(drift_area)//2:]
    smooth_drift[len(drift_area)//2:] = right_fit_values

    left_ewm = drift_area.iloc[::-1].ewm(span=5).mean()
    left_fit_values = left_ewm.iloc[0: len(drift_area // 2)]
    smooth_drift[0: len(drift_area // 2)] = left_fit_values[::-1]
    return smooth_drift


if __name__ == "__main__":
    slice_idx = [500]
    elec_df = pd.read_csv("./electricity-normalized.csv", nrows=1000,
                          usecols=["period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer", "class"])
    elec_df['class'] = elec_df['class'].apply(lambda element: 1 if element == "UP" else 0)
    reverse_elec_df = reverseSlice(elec_df, slice_idx, smooth=True)
