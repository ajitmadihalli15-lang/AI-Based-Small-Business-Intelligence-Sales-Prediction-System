from flask import Flask, render_template
from model import train_model, predict_sales

app = Flask(__name__)

model, df = train_model()


@app.route("/")
def dashboard():

    # Basic calculations
    total_sales = df["Sales"].sum()
    total_quantity = df["Quantity"].sum()
    total_orders = len(df)

    # Best-selling product
    product_sales = df.groupby("Product")["Sales"].sum()
    best_product = product_sales.idxmax()

    # Average sales
    average_sales = df["Sales"].mean()

    # AI prediction
    next_day = len(df) + 1
    predicted_sales = predict_sales(next_day, model)

    # Sales chart
    sales_dates = df["Date"].dt.strftime("%Y-%m-%d").tolist()
    sales_values = df["Sales"].tolist()

    # Product chart
    product_names = product_sales.index.tolist()
    product_values = product_sales.values.tolist()

    return render_template(
        "index.html",
        total_sales=total_sales,
        total_quantity=total_quantity,
        total_orders=total_orders,
        best_product=best_product,
        average_sales=average_sales,
        predicted_sales=predicted_sales,
        sales_dates=sales_dates,
        sales_values=sales_values,
        product_names=product_names,
        product_values=product_values
    )


if __name__ == "__main__":
    app.run(debug=True)