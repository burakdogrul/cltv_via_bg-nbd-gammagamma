#Importing required libraries

import datetime as dt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler
from lifetimes.plotting import plot_frequency_recency_matrix
from lifetimes.plotting import plot_probability_alive_matrix
from lifetimes.plotting import plot_period_transactions
from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases
from data_prep import *

import warnings
warnings.filterwarnings('ignore')

#Pandas set option
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

#Importing only 2009_2010 data
dataframe = pd.read_excel("online_retail_II_2009_2010")


dataframe.head()

#Backup dataframe
df = dataframe.copy()

check_df(df)

# We delete them incomplete information because it is customer-oriented
df.dropna(axis=0, inplace=True)

# The dataset has sales return invoice. We need to eliminate them
df = df[~df["Invoice"].str.contains("C", na=False)]

# As you can see, the negative values are available in price and quantity values.We need to eliminate them too.
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

# We need to push quantity and price values for better results
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

# Finally, we are creating a total price variable by multiplying the quantity and price values
df["TotalPrice"] = df["Quantity"] * df["Price"]

check_df(df)


def cltv_prediction(dataframe, month=3, plot=False):
    # We need to choose an analysis date. The latest data belongs to the following date and we choose 2 days after that date
    # df["InvoiceDate"].max() --> "2010-12-09"
    today_date = dt.datetime(2010, 12, 11)
    # We do grouping according to customer ids and calculate the following values
    # Recency--> Time passing over the customer's final purchase
    # T --> Time since the customer's first purchase. Age of customer for company. Tenure.
    # Frequency --> Frequency. The number of recurring sales.
    # Monetary --> Observed transaction value.
    cltv_df = dataframe.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                                    lambda date: (today_date - date.min()).days],
                                                    'Invoice': lambda num: num.nunique(),
                                                    'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

    # Calculate the frequency and recency values as weekly
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    # By definition, we choose customers greater than 1 frequency
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

    # Establishment of the BG-NBD model
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    # 3. Establishment of the GAMMA-GAMMA model
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    # Calculation of CLTV via BG-NBD and GG model.
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=month,
                                       freq="W",  # The frequency information of t.
                                       discount_rate=0.05)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

    # If scaler is wanted to use
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(cltv_final[["clv"]])
    cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])

    cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 5,
                                    labels=["about_to_sleep", "at_risk", "need_attention", "loyal_customers",
                                            "champions"])

    if plot:
        plot_probability_alive_matrix(bgf)

        plot_period_transactions(bgf)

    return cltv_final

cltv_6m = cltv_prediction(df, month=6 , plot=True)

cltv_6m.groupby("segment")["clv"].describe()