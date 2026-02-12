import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("superstore.csv", encoding="latin1")
data["Order Date"] = pd.to_datetime(data["Order Date"])
data = data.sort_values("Order Date")

daily_sales = data.groupby("Order Date")["Sales"].sum().reset_index()
daily_sales["date_num"] = daily_sales["Order Date"].map(pd.Timestamp.toordinal)
daily_sales["month"] = daily_sales["Order Date"].dt.month
daily_sales["dayofweek"] = daily_sales["Order Date"].dt.dayofweek

features = ["date_num", "month", "dayofweek"]

train = daily_sales.iloc[:-30]
test = daily_sales.iloc[-30:]

X_train = train[features]
y_train = train["Sales"]
X_test = test[features]
y_test = test["Sales"]

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

pred = model.predict(X_test_poly)

print(round(mean_absolute_error(y_test, pred), 2))

plt.figure(figsize=(10,5))
plt.plot(test["Order Date"], y_test, label="Actual")
plt.plot(test["Order Date"], pred, label="Predicted")
plt.legend()
plt.title("Sales Prediction")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

last_date = daily_sales["Order Date"].max()
future_dates = pd.date_range(last_date, periods=31)[1:]

future_df = pd.DataFrame()
future_df["Order Date"] = future_dates
future_df["date_num"] = future_df["Order Date"].map(pd.Timestamp.toordinal)
future_df["month"] = future_df["Order Date"].dt.month
future_df["dayofweek"] = future_df["Order Date"].dt.dayofweek

future_poly = poly.transform(future_df[features])
future_pred = model.predict(future_poly)

plt.figure(figsize=(10,5))
plt.plot(future_dates, future_pred)
plt.title("Future Sales Forecast (Next 30 Days)")
plt.xlabel("Date")
plt.ylabel("Predicted Sales")
plt.show()
