import pandas as pd
import numpy as np
from util import reverseSlice

class RealDriftGenerator:

    def __init__(self, df):
        self.origin_df = df
        self.reverse_df = pd.DataFrame(index=range(self.origin_df.shape[0]), columns=self.origin_df.columns
                                       , dtype=np.float32)
        self.drift_dict = {}

    def updateDrift(self, add_drift_dict):
        self.drift_dict = add_drift_dict
        return True

    def DriftSmooth(self, drift_area, drift_mode="middle"):
        smooth_drift = drift_area.copy()
        if drift_mode == "left" or drift_mode == "middle":
            right_ewm = drift_area.ewm(span=5).mean()
            right_fit_values = right_ewm.iloc[len(drift_area) // 2:]
            smooth_drift.iloc[len(drift_area) // 2:, :] = right_fit_values

        if drift_mode == "right" or drift_mode == "middle":
            left_ewm = drift_area[::-1].ewm(span=5).mean()
            left_fit_values = left_ewm.iloc[len(drift_area) // 2:]
            smooth_drift.iloc[0: len(drift_area) // 2, :] = left_fit_values[::-1]
        return smooth_drift

    def reverseSlice(self, drift_dict):
        self.updateDrift(drift_dict)
        slice_idx = list(self.drift_dict.keys())
        slice_idx.append(0)
        slice_idx.append(self.origin_df.shape[0])
        slice_idx.sort(reverse=True)
        head_pointer = 0

        for i in range(0, len(slice_idx) - 1):
            period_length = slice_idx[i] - slice_idx[i + 1]

            self.reverse_df.iloc[head_pointer:head_pointer + period_length, :] = self.origin_df.iloc[
                                                                                 slice_idx[i + 1]:slice_idx[i], :]
            head_pointer = head_pointer + period_length

            if i != 0 and i != self.origin_df.shape[0]:
                drift_width = self.drift_dict[slice_idx[i]][0]
                drift_mode = self.drift_dict[slice_idx[i]][1]
                reverse_slice_idx = self.origin_df.shape[0] - slice_idx[i]

                self.reverse_df.iloc[reverse_slice_idx - (drift_width // 2): reverse_slice_idx + (drift_width // 2),
                :] = self.DriftSmooth(
                    self.reverse_df.iloc[reverse_slice_idx - (drift_width // 2): reverse_slice_idx + (drift_width // 2), :],
                    drift_mode=drift_mode
                )

        return self.reverse_df.iloc[::-1].reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv("./electricity-normalized.csv",
                     usecols=["nswprice", "nswdemand", "vicprice", "vicdemand", "transfer", "class"],
                     nrows=1000)

    df["class"] = df["class"].apply(lambda element: 1 if element == "UP" else 0)
    drift_dict = {700: (100, "middle")}
    drift = RealDriftGenerator(df=df)
    reverse = drift.reverseSlice(drift_dict=drift_dict)

