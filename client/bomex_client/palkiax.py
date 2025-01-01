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


class Palkiax:
    """
    Model class holding a SARIMAX model
    - self.n : length of input weather observations
    - self.df : dataframe holding weather observations and data to plot
    - self.endog: pd.Series for endogenous data points (apparent temperature)
    - self.exog: pd.Series for concatenated exogenous data points (humidity, temperature)
    - self.results: SARIMAXResults containing fitted model's parameters and state
    """

    def __init__(self, source: str = WEATHER_PATH, train=False) -> None:
        df = pd.read_csv(source)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H%M%S")
        self.df = df.sort_values("timestamp")

        self.n: int = self.df.shape[0]

        if train:
            self.train_split: int = int(np.floor(self.n * TRAIN_SPLIT))
        else:
            self.train_split: int = self.n

        shifted_humidity = (
            self.df["humidity"].astype(float).shift(1)[1:].reset_index(drop=True)
        )
        shifted_temperature = (
            self.df["temperature"].astype(float).shift(1)[1:].reset_index(drop=True)
        )
        self.exog = pd.concat([shifted_humidity, shifted_temperature], axis=1)

        self.endog: pd.Series = (
            self.df["apparent_temperature"].astype(float)[:-1].reset_index(drop=True)
        )

    @property
    def observed_index(self):
        if self.results:
            return self.results.data.row_labels[-1]
        raise AttributeError("self.results not initialised yet")

    def train(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 48)) -> None:
        order = (1, 0, 1)
        seasonal_order = (1, 0, 1, 48)

        model = SARIMAX(
            endog=self.endog[: self.train_split],
            exog=self.exog[: self.train_split],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=True,
        )

        self.base_results = model.fit(disp=False)
        self.results = copy.deepcopy(self.base_results)
        with open("sarimax_base_model.pkl", "wb") as f:
            pickle.dump(self.base_results, f)

    def refit(self, endog_values: List[float], refit_parameters=False):
        for value in endog_values:
            self.endog.loc[self.observed_index] = value
            self.results = self.results.append(endog=value, refit=refit_parameters)

    def forecast(
        self, exog: pd.Series = None, steps: int = 1, risk_appetite: float = 0.05
    ):
        # Obtain step after last observed index
        lb = self.observed_index + 1
        ub = lb + steps
        required_range = pd.RangeIndex(lb, ub)
        missing_range = required_range.difference(self.exog.index)

        if not missing_range.empty:
            if not a.empty and len(exog) >= len(missing_range):  # not none
                self.exog = self.exog.append(exog.loc[missing_range]).sort_index()
            else:
                raise ValueError("Insufficient exog values provided")

        forecast = self.results.get_forecast(
            steps=steps, exog=self.exog[required_range]
        )
        confidence_interval = (
            forecast.conf_int(alpha=risk_appetite).iloc[steps - 1].to_dict()
        )
        predicted_value = forecast.predicted_mean.iloc[steps - 1]

        return {"value": predicted_value, "confidence": confidence_interval}

    def save_prediction_plot(self, ub, lb=1, dynamic=True, file="partial_sample.png"):
        if ub > self.observed_index:
            print(
                f"Insufficient endogenous data. Refit model first. Current step: {self.observed_index}"
            )
            return

        prediction = self.results.get_prediction(
            start=lb, end=ub, dynamic=dynamic, full_results=True
        )
        self.df["forecast"] = prediction.predicted_mean
        self.df["conf_ub"] = prediction.conf_int().iloc[:, 0]
        self.df["conf_lb"] = prediction.conf_int().iloc[:, 1]
        self.df["endog"] = self.endog
        plt.figure(figsize=(10, 5))
        plt.plot(self.df.iloc[lb:ub][["forecast", "endog", "conf_lb", "conf_ub"]])
        plt.legend(labels=["Forecast", "Actual", "Lower Bound", "Upper Bound"])
        plt.savefig(file)
        plt.show()
