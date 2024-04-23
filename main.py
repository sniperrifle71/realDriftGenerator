from RealDriftGenerator import RealDriftGenerator
import pandas as pd

df = pd.read_csv("../seattle-weather.csv", nrows=1000,
                 usecols=["precipitation", "temp_max", "temp_min", "wind", "weather"])
class_dict = {"drizzle": 0, "rain": 1, "sun": 2, "snow": 3, "fog": 4}
df["weather"] = df["weather"].apply(lambda x: class_dict[x])

feature_dim = df.shape[1] - 1
class_dim = int(df.iloc[:, feature_dim].max()) + 1
stream = RealDriftGenerator(df)
df = stream.reverseSlice(drift_dict={700: (100, "middle")})
df.to_csv("./weather_p700_w100_l1000.csv")