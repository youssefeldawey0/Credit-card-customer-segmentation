# Customer Segmentation Project

## What this project does

This project groups credit card customers according to their account and spending behavior. It uses KMeans to discover customer segments, then trains a Random Forest model to assign new customers to those same segments.

The dataset is `CC GENERAL.csv`. It contains customer behavior data, but it does not contain a fraud label. **This project does not detect fraud and cannot tell whether a transaction is fraudulent or legitimate.**

## What do labels such as 0 and 1 mean?

`0` and `1` are segment IDs created by KMeans. They identify different groups of customers; they do not mean “fraud” or “not fraud.” The IDs themselves do not describe the groups. To understand a segment, review its average feature values in `customer_cluster_profiles.csv`.

## How can the segments be useful?

The segments can help a bank or business:

- Compare customer behavior, such as purchases, balances, and cash advances.
- Design offers and services for different behavior patterns.
- Plan marketing campaigns for groups with similar activity.
- Explore which groups may benefit from payment or installment options.

These are behavioral groups, not guaranteed predictions about what a customer will do.

## Run the Streamlit app

Install the required packages:

```powershell
pip install -r requirements.txt
```

Start the app:

```powershell
streamlit run app.py
```

Upload a customer CSV with the same feature columns as `CC GENERAL.csv`, or use the project dataset. Select **Assign customer segments** to see the predicted segment IDs and download the results as a CSV.

## Project files

- `fraud detetct-1.ipynb` — notebook with the data exploration, preprocessing, PCA, KMeans segmentation, and classifier pipeline.
- `app.py` — Streamlit interface for assigning and downloading customer segments.
- `CC GENERAL.csv` — customer behavior dataset used by the notebook and app.
- `customer_cluster_classifier.joblib` — saved model used by the app.
- `customer_cluster_profiles.csv` — average feature values for each segment in the training dataset.
- `requirements.txt` — Python packages needed to run the app.
