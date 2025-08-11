import streamlit as st
import pyodbc
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# ===================== SQL Connection Function =====================
def load_data():
    SERVER = "172.27.6.3"
    DATABASE = "PO_Database"
    USERNAME = "CMS_LAB"
    PASSWORD = "CMS@LAB22"

    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={SERVER};"
        f"DATABASE={DATABASE};"
        f"UID={USERNAME};"
        f"PWD={PASSWORD}"
    )

    query = """
    SELECT
        OrderDate,
        Quantity,
        MaterialID,
        SupplierID
    FROM PurchaseOrders
    ORDER BY OrderDate
    """
    df = pd.read_sql(query, conn)
    conn.close()

    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    return df

# ===================== Streamlit App =====================
st.set_page_config(page_title="SQL Forecast Dashboard", layout="wide")

st.title("📊 SQL Forecast Dashboard")
st.markdown("Select filters and see forecasted quantities.")

# Load data
df = load_data()

# Sidebar Filters
material_list = ["All"] + sorted(df["MaterialID"].unique().tolist())
supplier_list = ["All"] + sorted(df["SupplierID"].unique().tolist())

material_filter = st.sidebar.selectbox("Select Material", material_list)
supplier_filter = st.sidebar.selectbox("Select Supplier", supplier_list)

# Apply filters
filtered_df = df.copy()
if material_filter != "All":
    filtered_df = filtered_df[filtered_df["MaterialID"] == material_filter]
if supplier_filter != "All":
    filtered_df = filtered_df[filtered_df["SupplierID"] == supplier_filter]

# Check if data available
if filtered_df.empty:
    st.warning("No data available for selected filters.")
else:
    # Prepare daily data
    daily_data = filtered_df.groupby("OrderDate")["Quantity"].sum().reset_index()
    forecast_df = daily_data.rename(columns={"OrderDate": "ds", "Quantity": "y"})

    # Forecast
    model = Prophet()
    model.fit(forecast_df)
    future = model.make_future_dataframe(periods=180)  # 6 months
    forecast = model.predict(future)

    # Show data
    st.subheader("📅 Forecast Data")
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10))

    # Plot
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Download option
    csv = forecast[["ds", "yhat"]].to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name="forecast_results.csv",
        mime="text/csv"
    )
