from featexp import *
import pandas as pd


data_folder = "article/home-credit-default-risk/"
data_file = "application_test.csv"
df = pd.read_csv(data_folder + data_file)


data_train = df
# Plots drawn for all features if nothing is passed in feature_list parameter.
get_univariate_plots(data=data_train, target_col='AMT_CREDIT', features_list=['DAYS_BIRTH'], bins=10)

stats = get_trend_stats(data=data_train, target_col='AMT_CREDIT', data_test=data_train)
print(stats)

get_univariate_plots(data=data_train, target_col='AMT_CREDIT', data_test=data_train, features_list=['DAYS_EMPLOYED'])