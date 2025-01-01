import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import matplotlib.pyplot as plt
import copy
from typing import List, Union

WEATHER_PATH = "../weather_observations.csv"
TRAIN_SPLIT = 0.75


class Palkia:
    """
    - self.n
    - self.df : dataframe
    - self.endog
    - self.results: SARIMAX Results
    - self.steps : holds the latest step in the time series which has been trained on observed endogenous data
    """

    def __init__(self, source: str = WEATHER_PATH, train=False) -> None:
        df = pd.read_csv(source)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H%M%S")
        self.df = df.sort_values("timestamp").set_index("timestamp")

        self.df["predictions"] = self.df["apparent_temperature"]
        self.n: int = self.df.shape[0]

        if train:
            self.train_split: int = int(np.floor(self.n * TRAIN_SPLIT))
        else:
            self.train_split: int = self.n

        self.steps: int = self.train_split

        self.endog: pd.Series = self.df["apparent_temperature"].astype(float)

    def train(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 48)) -> None:
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 48)

        model = SARIMAX(
            endog=self.endog[: self.train_split],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=True,
        )

        self.base_results = model.fit(disp=False)
        self.results = copy.deepcopy(self.base_results)
        with open("base_model.pkl", "wb") as f:
            pickle.dump(self.base_results, f)

    def refit(self, endog_values: List[float], refit_parameters=False):
        for value in endog_values:
            self.endog.loc[self.steps] = value
            self.results = self.results.append(endog=value, refit=refit_parameters)
            self.steps += 1

    def forecast(self, steps: int = 1, risk_appetite: float = 0.05):
        forecast = self.results.get_forecast(steps=steps)
        confidence_interval = (
            forecast.conf_int(alpha=risk_appetite).iloc[steps - 1].to_dict()
        )
        predicted_value = forecast.predicted_mean.iloc[steps - 1]

        return {"value": predicted_value, "confidence": confidence_interval}

    def save_prediction_plot(
        self, lb, ub, file1="full_sample.png", file2="partial_sample.png"
    ):
        if ub > self.steps:
            print(
                f"Insufficient endogenous data. Refit model first. Current step: {self.steps}"
            )
            return

        prediction = self.results.get_prediction(
            start=lb, end=ub, dynamic=True, full_results=True
        )
        self.df["forecast"] = prediction.predicted_mean
        self.df["lb"] = prediction.conf_int().iloc[:, 0]
        self.df["ub"] = prediction.conf_int().iloc[:, 1]

        self.df[["forecast"]].plot()
        self.endog.plot()
        plt.savefig(file1)
        plt.close()

        self.df.iloc[lb:ub][["forecast", "lb", "ub"]].plot(figsize=(10, 5))
        self.endog.iloc[lb:ub].plot()
        plt.savefig(file2)
        plt.show()
