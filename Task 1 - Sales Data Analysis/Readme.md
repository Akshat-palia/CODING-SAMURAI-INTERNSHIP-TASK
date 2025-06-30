# Task 1 - Sales Data Analysis

## 📊 Project Overview
This project performs Exploratory Data Analysis (EDA) on a retail sales dataset. The goal is to uncover patterns, trends, and insights from the sales data that can assist in making data-driven business decisions.

## 🗂️ Project Structure
Task 1 - Sales Data Analysis/
|
|-- data/
| |-- sales_data_sample.csv # Raw dataset
|
|-- outputs/
| |-- screenshots_of_visualizations/ # Screenshots of graphs and charts
| |-- sales_summary.xlsx # Excel summary output (optional)
|
|-- Sales_Data_Analysis.ipynb # Jupyter Notebook containing the code
|-- README.md # Project documentation


## 🚀 Skills & Tools Used
- Python
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook

## 🔧 Data Cleaning
- Converted `ORDERDATE` to datetime format.
- Dropped columns with high null values (`ADDRESSLINE2`).
- Handled missing values in `STATE`, `POSTALCODE`, and `TERRITORY`.
- Verified no duplicate entries.

## 📈 Exploratory Data Analysis

### ✅ **Key Insights**
- **Order Status:** Majority of orders are 'Shipped'.
- **Top Countries by Orders:** USA leads, followed by Spain and France.
- **Deal Size Distribution:** Medium deals dominate, followed by small and large.
- **Top Product Line:** Classic Cars has the highest sales.
- **Top Customers:** Euro+ Shopping Channel, Mini Gifts Distributors Ltd., and Australian Collectors, Co.
- **Peak Sales Period:** Q4 (Oct-Dec), especially in November and December.

### 📊 **Visualizations Created**
- Order Status Distribution
- Orders by Country
- Deal Size Distribution
- Sales Trend over Months and Years
- Sales by Product Line
- Sales by Customer

## 📜 Conclusion
This project demonstrates how basic data analysis and visualization techniques can yield valuable business insights. It provides an overview of customer behavior, product performance, and temporal sales trends.

## ✍️ Author
**Akshat Palia**

---
