#AirPassengers Dataset for Time Series Analysis

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# 1. Load AirPassengers (monthly totals, Jan 1949 - Dec 1960)
values = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
]
AirPassengers = pd.Series(values, index=pd.date_range("1949-01", periods=144, freq="MS"),
                          name="AirPassengers")
print("1. AirPassengers loaded:", AirPassengers.shape, "\n")

# 2. First five and last five observations
print("2. First five\n", AirPassengers.head(5), "\n\n   Last five\n", AirPassengers.tail(5), "\n")

# 3. Start and end
print("3. Start:", AirPassengers.index.min().strftime("%b %Y"),
      "| End:", AirPassengers.index.max().strftime("%b %Y"), "\n")

# 4. Frequency and time-series characteristics
print("4. Frequency (observations per year): 12  (index freq =", AirPassengers.index.freqstr, ")")
print("   Type:", type(AirPassengers).__name__, "| dtype:", AirPassengers.dtype)
print("   Mean = %.2f, Std = %.2f, Min = %d, Max = %d\n" %
      (AirPassengers.mean(), AirPassengers.std(), AirPassengers.min(), AirPassengers.max()))

# 5. Monthly trend plot
plt.figure(figsize=(11, 5))
plt.plot(AirPassengers, color="navy")
plt.title("5. Monthly Airline Passengers (1949-1960)")
plt.xlabel("Year"); plt.ylabel("Passengers (thousands)")
plt.grid(True, alpha=0.4)
plt.tight_layout(); plt.show()

# 6. Decompose into trend, seasonal, random
#    (multiplicative, because seasonal swings grow with the level of the series)
decomp = seasonal_decompose(AirPassengers, model="multiplicative", period=12)
print("6. Decomposition (first rows)\n",
      pd.DataFrame({"trend": decomp.trend, "seasonal": decomp.seasonal,
                    "random": decomp.resid}).head(12).round(3), "\n")

# 7. Plot the decomposed components
fig = decomp.plot()
fig.set_size_inches(11, 8)
fig.suptitle("7. Decomposition of AirPassengers (multiplicative)", y=1.01)
plt.tight_layout(); plt.show()

# 8. ACF and PACF
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
plot_acf(AirPassengers, lags=36, ax=ax[0]);  ax[0].set_title("8a. ACF")
plot_pacf(AirPassengers, lags=36, ax=ax[1], method="ywm"); ax[1].set_title("8b. PACF")
plt.tight_layout(); plt.show()
print("8. ACF decays slowly (non-stationary, strong trend) with peaks at lags 12, 24, 36 "
      "(yearly seasonality). PACF cuts off after the first lags.\n")

# 9. Fit an ARIMA model  (seasonal ARIMA (2,1,1)(0,1,0)[12] on the raw series)
model = ARIMA(AirPassengers, order=(2, 1, 1), seasonal_order=(0, 1, 0, 12))
fit = model.fit()

# 10. Model summary + interpretation
print("10. ARIMA model summary")
print(fit.summary())
print("""
Interpretation:
 - order (2,1,1): 2 autoregressive terms, 1 non-seasonal difference, 1 moving-average term.
 - seasonal (0,1,0,12): one seasonal difference (lag 12) removes the yearly pattern.
 - Coefficients with small p-values (P>|z|) are significant; lower AIC/BIC = better model.
 - Ljung-Box p > 0.05 means residuals look like white noise (no leftover autocorrelation).
""")

# 11. Forecast the next 12 months
forecast = fit.get_forecast(steps=12)
pred = forecast.predicted_mean
conf = forecast.conf_int()
print("11. 12-month forecast\n", pred.round(1), "\n")

# 12. Actual vs forecast plot
plt.figure(figsize=(11, 5))
plt.plot(AirPassengers, label="Actual", color="navy")
plt.plot(pred, label="Forecast", color="red", linestyle="--")
plt.fill_between(conf.index, conf.iloc[:, 0], conf.iloc[:, 1], color="red", alpha=0.15,
                 label="95% confidence interval")
plt.title("12. AirPassengers: Actual vs 12-month ARIMA Forecast")
plt.xlabel("Year"); plt.ylabel("Passengers (thousands)")
plt.legend(); plt.grid(True, alpha=0.4)
plt.tight_layout(); plt.show()
