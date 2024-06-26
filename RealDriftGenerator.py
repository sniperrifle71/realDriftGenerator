import pandas as pd
import numpy as np


class RealDriftGenerator:
    """
    RealDriftGenerator generates concept drift with user-defined position, width in target source time-series dataset
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        :param df: source time-series dataset
        """
        self.origin_df = df
        self.reverse_df = pd.DataFrame(index=range(self.origin_df.shape[0]), columns=self.origin_df.columns
                                       , dtype=np.float32)
        self.drift_dict = {}

    def DriftSmooth(self, drift_area: pd.DataFrame, drift_mode="middle") -> pd.DataFrame:
        """
        Smooth the concept drift with a width of 1 to the user defined drift width

        :param drift_area: expected drift area
        :param drift_mode: Just use middle
        :return:smoothed expected drift area
        """
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

    def reverseSlice(self, drift_dict: dict) -> pd.DataFrame:
        """
        Generate concept drift in the source time series dataset
        :param drift_dict: {position:(width, drift_mode)}, contains user-defined drift details
        :return: time series dataset with concept drift.
        """
        self.drift_dict = drift_dict
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
                    self.reverse_df.iloc[reverse_slice_idx - (drift_width // 2): reverse_slice_idx + (drift_width // 2),
                    :],
                    drift_mode=drift_mode
                )

        return self.reverse_df.iloc[::-1].reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv("../seattle-weather.csv", nrows=1000,
                     usecols=["precipitation", "temp_max", "temp_min", "wind", "weather"])
    class_dict = {"drizzle": 0, "rain": 1, "sun": 2, "snow": 3, "fog": 4}
    df["weather"] = df["weather"].apply(lambda x: class_dict[x])

    feature_dim = df.shape[1] - 1
    class_dim = int(df.iloc[:, feature_dim].max()) + 1
    stream = RealDriftGenerator(df)
    df = stream.reverseSlice(drift_dict={700: (100, "middle")})
    df.to_csv("./weather_p700_w100_l1000.csv")
