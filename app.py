
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"

# --------------------------------------------------
# 3. LOAD REPORTS
# --------------------------------------------------

@st.cache_data
def load_report(filename):
    file_path = REPORT_DIR / filename

    if not file_path.exists():
        return None

    try:
        data = pd.read_csv(file_path)

        if "date" in data.columns:
            data["date"] = pd.to_datetime(
                data["date"],
                errors="coerce"
            )

        return data

    except Exception as error:
        st.error(f"Error loading {filename}: {error}")
        return None


forecast_df = load_report("16_day_demand_forecast.csv")
daily_df = load_report("daily_forecast_summary.csv")
store_df = load_report("store_demand_forecast.csv")
category_df = load_report("category_demand_forecast.csv")
inventory_df = load_report("inventory_optimization_results.csv")
store_inventory_df = load_report("store_inventory_summary.csv")
category_inventory_df = load_report("category_inventory_summary.csv")

model_df = None

for filename in [
    "final_test_comparison.csv",
    "final_algorithm_comparison.csv",
    "algorithm_comparison.csv",
    "model_comparison.csv"
]:
    model_df = load_report(filename)

    if model_df is not None:
        break

# --------------------------------------------------
# 4. APPLICATION HEADER
# --------------------------------------------------

st.title("📦 Supply Chain Intelligence Dashboard")

st.markdown(
    "### Demand Forecasting | Inventory Optimization | "
    "Business Analytics"
)

st.caption(
    "Corporación Favorita Grocery Sales Forecasting Project"
)

st.divider()

# --------------------------------------------------
# 5. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Dashboard",
    [
        "Executive Overview",
        "Demand Forecasting",
        "Store Analysis",
        "Product Analysis",
        "Inventory Optimization",
        "Model Performance",
        "Data Explorer"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("Dashboard Filters")

# --------------------------------------------------
# 6. GLOBAL FORECAST FILTERS
# --------------------------------------------------

filtered_forecast = None

if forecast_df is not None and not forecast_df.empty:

    filtered_forecast = forecast_df.copy()

    if "date" in filtered_forecast.columns:

        valid_dates = filtered_forecast["date"].dropna()

        if not valid_dates.empty:

            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()

            selected_dates = st.sidebar.date_input(
                "Forecast date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            if isinstance(selected_dates, (tuple, list)):
                if len(selected_dates) == 2:
                    start_date, end_date = selected_dates
                    filtered_forecast = filtered_forecast[
                        filtered_forecast["date"].dt.date.between(
                            start_date,
                            end_date
                        )
                    ]
            else:
                filtered_forecast = filtered_forecast[
                    filtered_forecast["date"].dt.date
                    == selected_dates
                ]

    if "store_nbr" in filtered_forecast.columns:

        stores = sorted(
            filtered_forecast["store_nbr"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_store = st.sidebar.selectbox(
            "Select Store",
            ["All Stores"] + stores
        )

        if selected_store != "All Stores":
            filtered_forecast = filtered_forecast[
                filtered_forecast["store_nbr"]
                == selected_store
            ]

    if "family" in filtered_forecast.columns:

        categories = sorted(
            filtered_forecast["family"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_category = st.sidebar.selectbox(
            "Product Category",
            ["All Categories"] + categories
        )

        if selected_category != "All Categories":
            filtered_forecast = filtered_forecast[
                filtered_forecast["family"]
                == selected_category
            ]

# --------------------------------------------------
# 7. EXECUTIVE OVERVIEW
# --------------------------------------------------

if page == "Executive Overview":

    st.header("Executive Overview")

    if filtered_forecast is None or filtered_forecast.empty:

        st.warning(
            "Forecast report is missing or contains no data. "
            "Generate the Phase 7 forecast first."
        )

    else:

        total_demand = filtered_forecast[
            "predicted_sales"
        ].sum()

        total_stores = (
            filtered_forecast["store_nbr"].nunique()
        )

        total_categories = (
            filtered_forecast["family"].nunique()
        )

        total_days = (
            filtered_forecast["date"].nunique()
        )

        daily_total = (
            filtered_forecast
            .groupby("date")["predicted_sales"]
            .sum()
        )

        average_daily_demand = daily_total.mean()

        # KPI cards
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Predicted Demand",
            f"{total_demand:,.0f}"
        )

        col2.metric(
            "Average Daily Demand",
            f"{average_daily_demand:,.0f}"
        )

        col3.metric(
            "Active Stores",
            f"{total_stores:,}"
        )

        col4.metric(
            "Forecast Days",
            total_days
        )

        st.divider()

        # Daily demand trend
        daily_chart = (
            filtered_forecast
            .groupby("date", as_index=False)
            ["predicted_sales"]
            .sum()
        )

        fig = px.line(
            daily_chart,
            x="date",
            y="predicted_sales",
            markers=True,
            title="Daily Predicted Grocery Demand"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Category demand
        category_chart = (
            filtered_forecast
            .groupby("family", as_index=False)
            ["predicted_sales"]
            .sum()
            .sort_values(
                "predicted_sales",
                ascending=False
            )
            .head(10)
        )

        fig2 = px.bar(
            category_chart,
            x="family",
            y="predicted_sales",
            title="Top 10 Product Categories by Demand"
        )

        st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# 8. DEMAND FORECASTING
# --------------------------------------------------

elif page == "Demand Forecasting":

    st.header("Demand Forecasting")

    if filtered_forecast is None or filtered_forecast.empty:

        st.warning("Forecast data is unavailable.")

    else:

        daily_demand = (
            filtered_forecast
            .groupby("date", as_index=False)
            ["predicted_sales"]
            .sum()
        )

        fig = px.line(
            daily_demand,
            x="date",
            y="predicted_sales",
            markers=True,
            title="Demand Forecast Trend"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast Details")

        st.dataframe(
            filtered_forecast,
            use_container_width=True,
            hide_index=True
        )

        csv_data = filtered_forecast.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Forecast CSV",
            csv_data,
            "demand_forecast.csv",
            "text/csv"
        )

# --------------------------------------------------
# 9. STORE ANALYSIS
# --------------------------------------------------

elif page == "Store Analysis":

    st.header("Store-wise Demand Analysis")

    if filtered_forecast is None or filtered_forecast.empty:

        st.warning("Forecast data is unavailable.")

    else:

        store_analysis = (
            filtered_forecast
            .groupby("store_nbr", as_index=False)
            ["predicted_sales"]
            .sum()
            .sort_values(
                "predicted_sales",
                ascending=False
            )
        )

        fig = px.bar(
            store_analysis,
            x="store_nbr",
            y="predicted_sales",
            title="Store-wise Predicted Demand"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Store Performance")

        st.dataframe(
            store_analysis,
            use_container_width=True,
            hide_index=True
        )

# --------------------------------------------------
# 10. PRODUCT CATEGORY ANALYSIS
# --------------------------------------------------

elif page == "Product Analysis":

    st.header("Product Category Analysis")

    if filtered_forecast is None or filtered_forecast.empty:

        st.warning("Forecast data is unavailable.")

    else:

        product_analysis = (
            filtered_forecast
            .groupby("family", as_index=False)
            ["predicted_sales"]
            .sum()
            .sort_values(
                "predicted_sales",
                ascending=False
            )
        )

        fig = px.bar(
            product_analysis.head(20),
            x="predicted_sales",
            y="family",
            orientation="h",
            title="Top 20 Product Categories"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            product_analysis,
            use_container_width=True,
            hide_index=True
        )

# --------------------------------------------------
# 11. INVENTORY OPTIMIZATION
# --------------------------------------------------

elif page == "Inventory Optimization":

    st.header("Inventory Optimization")

    if inventory_df is None:

        st.warning(
            "Inventory report not found. "
            "Complete Phase 8 and generate "
            "inventory_optimization_results.csv."
        )

    else:

        inv = inventory_df.copy()

        # Apply selected store/category filters
        if (
            filtered_forecast is not None
            and "store_nbr" in inv.columns
        ):

            available_keys = filtered_forecast[
                ["store_nbr", "family"]
            ].drop_duplicates()

            inv = inv.merge(
                available_keys,
                on=["store_nbr", "family"],
                how="inner"
            )

        if not inv.empty:

            order_qty = (
                pd.to_numeric(
                    inv["recommended_order_qty"],
                    errors="coerce"
                )
                if "recommended_order_qty" in inv.columns
                else pd.Series(dtype=float)
            )

            valid_order_qty = order_qty.dropna()

            if not valid_order_qty.empty:

                col1, col2 = st.columns(2)

                col1.metric(
                    "Recommended Order Quantity",
                    f"{valid_order_qty.sum():,.0f}"
                )

                col2.metric(
                    "Inventory Combinations",
                    f"{len(inv):,}"
                )

            else:

                st.info(
                    "Enter actual on-hand and on-order "
                    "inventory values in your Phase 8 "
                    "inventory input template and rerun "
                    "the inventory calculations."
                )

            if "inventory_risk" in inv.columns:

                risk_counts = (
                    inv["inventory_risk"]
                    .value_counts()
                    .reset_index()
                )

                risk_counts.columns = [
                    "Risk",
                    "Count"
                ]

                fig = px.bar(
                    risk_counts,
                    x="Risk",
                    y="Count",
                    title="Inventory Risk Overview"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            st.subheader("Inventory Recommendations")

            st.dataframe(
                inv,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Inventory Report",
                inv.to_csv(index=False).encode("utf-8"),
                "inventory_report.csv",
                "text/csv"
            )

        else:

            st.info("No inventory records match the selected filters.")

# --------------------------------------------------
# 12. MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.header("Machine Learning Model Performance")

    if model_df is None:

        st.warning(
            "Model comparison report not found. "
            "Generate your model comparison report "
            "from Phase 6."
        )

    else:

        st.subheader("Model Comparison")

        st.dataframe(
            model_df,
            use_container_width=True,
            hide_index=True
        )

        if {
            "Model",
            "WAPE (%)"
        }.issubset(model_df.columns):

            fig = px.bar(
                model_df.sort_values("WAPE (%)"),
                x="Model",
                y="WAPE (%)",
                title="Model Comparison - WAPE"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        metric = st.selectbox(
            "Select Evaluation Metric",
            [
                col for col in [
                    "MAE",
                    "RMSE",
                    "RMSLE",
                    "WAPE (%)"
                ]
                if col in model_df.columns
            ]
        )

        if metric:

            fig2 = px.bar(
                model_df.sort_values(metric),
                x="Model",
                y=metric,
                title=f"Model Comparison - {metric}"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

# --------------------------------------------------
# 13. DATA EXPLORER
# --------------------------------------------------

elif page == "Data Explorer":

    st.header("Data Explorer")

    available_reports = {
        "Demand Forecast": forecast_df,
        "Daily Summary": daily_df,
        "Store Summary": store_df,
        "Category Summary": category_df,
        "Inventory Details": inventory_df,
        "Store Inventory": store_inventory_df,
        "Category Inventory": category_inventory_df,
        "Model Comparison": model_df
    }

    available_names = [
        name for name, data in available_reports.items()
        if data is not None
    ]

    if available_names:

        selected_report = st.selectbox(
            "Select Report",
            available_names
        )

        selected_data = available_reports[
            selected_report
        ]

        st.write("Dataset Shape:", selected_data.shape)

        st.dataframe(
            selected_data.head(1000),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Selected Dataset",
            selected_data.to_csv(
                index=False
            ).encode("utf-8"),
            f"{selected_report.lower().replace(' ', '_')}.csv",
            "text/csv"
        )

    else:

        st.warning("No reports are available.")

# --------------------------------------------------
# 14. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Supply Chain Intelligence | "
    "Demand Forecasting & Inventory Analytics"
)
