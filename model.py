import pandas as pd
from sklearn.linear_model import LinearRegression

def train_model():

    df = pd.read_csv("sales_data.csv")

    df["Date"] = pd.to_datetime(df["Date"])

    # Calculate total sales
    df["Sales"] = df["Quantity"] * df["Price"]

    # Convert date to numerical value
    df["Day"] = (df["Date"] - df["Date"].min()).dt.days

    X = df[["Day"]]
    y = df["Sales"]

    model = LinearRegression()
    model.fit(X, y)

    return model, df


def predict_sales(days, model):

    prediction = model.predict([[days]])

    return round(prediction[0], 2)