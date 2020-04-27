import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv("../input/consumeranalytics/complaints-wf-updated.csv",low_memory=False)
df.head()
df.shape
df.dtypes
df.isnull().sum().sort_values(ascending=False)
p_product_discussions = round(df["product"].value_counts() / len(df["product"]) * 100,2)
p_product_discussions
plt.figure(figsize=(15,5))
sns.countplot(df['product'])
plt.show()
temp = df.company.value_counts()[:10]
temp
temp = df.state.value_counts()[:10]
temp
plt.figure(figsize=(15,5))
sns.barplot(temp.index,temp.values)
temp = df.company_response_to_consumer.value_counts()
temp
plt.figure(figsize=(15,5))
sns.barplot(y = temp.index, x= temp.values)
df.timely_response.value_counts()
sns.countplot(df.timely_response)
df['consumer_disputed?'].value_counts()
sns.countplot(df['consumer_disputed?'])
df['date_received'] = pd.to_datetime(df['date_received'])
df['year_received'], df['month_received'] = df['date_received'].dt.year, df['date_received'].dt.month
df.head()
df.year_received.value_counts()
sns.countplot(df.year_received)
df.month_received.value_counts()
sns.countplot(df.month_received)
