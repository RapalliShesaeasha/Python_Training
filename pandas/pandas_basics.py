import pandas as pd


def pandas_operations():
    # Read CSV
    sales_df = pd.read_csv("sales.csv")
    print("Sales Data:\n", sales_df)

    # Read Excel
    inventory_df = pd.read_excel("inventory.xlsx")
    print("Inventory Data:\n", inventory_df)

    # Add total amount column
    sales_df["total_amount"] = sales_df["price"] * sales_df["quantity"]

    # Filter high value orders
    high_value_orders = sales_df[sales_df["total_amount"] > 200]
    print("High Value Orders:\n", high_value_orders)

    # Group by category
    revenue_by_category = sales_df.groupby("category")["total_amount"].sum()
    print("Revenue by Category:\n", revenue_by_category)

    # Merge sales with inventory
    merged_df = pd.merge(sales_df, inventory_df, on="product")
    print("Merged Data:\n", merged_df)


if __name__ == "__main__":
    pandas_operations()
