"""Python code extracted from the provided HTML notebook export.
Review dataset paths and notebook-only commands before running as a script.
"""

# SI 671 Project: E-commerce User Behavior Analysis and Modeling
# 
# Xinyu Zhang (xyuzhang, 42192372)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
%matplotlib inline
from matplotlib.patches import Shadow
from matplotlib.lines import Line2D

import warnings
warnings.filterwarnings("ignore")

# 1. Data Preprocessing

from sklearn.model_selection import train_test_split

# 1.1 Data Sampling

column_names = ["User_ID", "Product_ID", "Category_ID", "Behavior", "Timestamp"]
df_ori = pd.read_csv("UserBehavior.csv", names=column_names)

print("There are in total " + str(len(df_ori)) + " records in the original dataset.")
print("There are in total " + str(df_ori['User_ID'].nunique()) + " unique users in the original dataset.") 
print("There are in total " + str(df_ori['Category_ID'].nunique()) + " categories in the original dataset.") 
print("There are in total " + str(df_ori['Product_ID'].nunique()) + " products in the original dataset.")

# sample users
users = df_ori['User_ID'].unique()
_, users_sampled = train_test_split(users, test_size=0.05, random_state=42)
df_5M = df_ori[df_ori['User_ID'].isin(users_sampled)]
df_5M.head()

print("There are in total " + str(len(df_5M)) + " records in the sampled dataset.")
print("There are in total " + str(df_5M['User_ID'].nunique()) + " unique users in the sampled dataset.") 
print("There are in total " + str(df_5M['Category_ID'].nunique()) + " categories in the sampled dataset.") 
print("There are in total " + str(df_5M['Product_ID'].nunique()) + " products in the sampled dataset.")

df_5M.to_csv("UserBehavior_5M.csv")

# after the first time initialization, to reproduce the result, use the code
# df_5M = pd.read_csv("UserBehavior_5M.csv")

# 1.2 Data Processing
# 
# - Data Cleaning

# check if there are null values
df_5M.isnull().sum()

# check if there are duplicated records
df_5M.duplicated().sum()

df_5M.drop_duplicates(inplace=True)
df_5M.duplicated().sum()

# rename the "Behavior" column
behavior_mapping = {'pv': 'PageView', 'buy': 'Buy', 'cart': 'AddToCart', 'fav': 'Favorite'}
df_5M['Behavior'] = df_5M['Behavior'].replace(behavior_mapping)
df_5M.head()

# - Format Transformation

# transform `timestamp` to datetime
df_5M['Datetime'] = pd.to_datetime(df_5M['Timestamp'], unit='s')
df_5M.head()

# Filter out the valid year range

df_5M['Datetime'].dt.year.value_counts()

df_5M = df_5M[df_5M['Datetime'].dt.year == 2017]
df_5M['Datetime'].dt.year.unique()

df_5M['Datetime'].dt.date.value_counts()[:15]

df_5M = df_5M[(df_5M['Datetime'] >= '2017-11-25 00:00:00') & (df_5M['Datetime'] <= '2017-12-03 23:59:59')]

df_5M['Day_of_Week'] = df_5M['Datetime'].dt.day_name()
df_5M['Hour'] = df_5M['Datetime'].dt.hour
df_5M['Date'] = df_5M['Datetime'].dt.date
df_5M.head()

df_5M['User_ID'].nunique()

len(df_5M)

# df_5M.to_csv("UserBehavior_5M_cleaned.csv")

# 2. Exploratory Data Analysis

# The top 5 most popular categories
df_5M['Category_ID'].value_counts()[:5]

# Daily Count of "Buy" Behavior over Time
buy = df_5M[df_5M.Behavior == 'Buy']
buy_count = buy.groupby('Date').size().reset_index(name='count')
buy_count = buy_count.sort_values('Date')

# Day_of_week count of "Buy" Behavior over Time
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
buy_day_of_week = buy.groupby('Day_of_Week').size().reset_index(name='count')
# used for ordinal data where the order matters
buy_day_of_week['Day_of_Week'] = pd.Categorical(buy_day_of_week['Day_of_Week'], categories=days_order, ordered=True)  
buy_day_of_week = buy_day_of_week.sort_values('Day_of_Week')
buy_day_of_week

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
ax2.plot(buy_count['Date'], buy_count['count'])
ax2.set_xlabel('Date', fontsize=18)
ax2.set_ylabel('Daily Count', fontsize=18)
ax2.tick_params(axis='x', rotation=45)
ax1.plot(buy_day_of_week['Day_of_Week'], buy_day_of_week['count'])
ax1.set_xlabel('Day of Week', fontsize=18)
ax1.set_ylabel('Count of "Buy"', fontsize=18)
ax1.tick_params(axis='x', rotation=45)
f.suptitle('Counts of "Buy" Behavior over Time', fontsize=20) 
plt.subplots_adjust(wspace=0.3)
plt.savefig('Buy_Count.png', format='png', bbox_inches='tight')

df_5M['Day_of_Week'] = pd.Categorical(df_5M['Day_of_Week'], categories=days_order, ordered=True)  
df_5M = df_5M.sort_values('Day_of_Week')
df_5M

hourly_behavior = df_5M.groupby(['Hour', 'Behavior']).size().reset_index(name='Behavior_Count')
hourly_behavior_count = hourly_behavior.pivot_table(index='Hour', columns='Behavior', values='Behavior_Count', fill_value=0)
hourly_behavior_count.reset_index(inplace=True)
hourly_behavior_count.head()

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
ax1.plot(buy_count['Date'], buy_count['count'])
ax1.set_xlabel('Date', fontsize=18)
ax1.set_ylabel('Daily Count', fontsize=18)
ax1.tick_params(axis='x', rotation=45)
ax2.plot(buy_day_of_week['Day_of_Week'], buy_day_of_week['count'])
ax2.set_xlabel('Day of Week', fontsize=18)
ax2.set_ylabel('Count of "Buy"', fontsize=18)
ax2.tick_params(axis='x', rotation=45)
f.suptitle('Counts of "Buy" Behavior over Time', fontsize=20) 
plt.subplots_adjust(wspace=0.3)
plt.savefig('Buy_Count.png', format='png', bbox_inches='tight')

plt.figure(1, figsize=(15, 6))
n = 0 
for x in ['Hour', 'Day_of_Week']:
    n += 1
    plt.subplot(1, 2, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.2)
    sns.countplot(data=df_5M, x=x)
    plt.title('Countplot of {}'.format(x), fontsize=18)

plt.savefig('Countplot of Hour and Day_of_Week.png', format='png', bbox_inches='tight')
plt.show()

df_5M.groupby('Hour')['Behavior'].count().sum()

# Calculate conversion rate for each category
conversion = df_5M[df_5M['Behavior'].isin(['Buy', 'PageView'])]
conversion_pb = conversion.groupby('Category_ID')['Behavior'].value_counts().unstack().fillna(0)
conversion_pb = conversion_pb[conversion_pb['PageView'] >= conversion_pb['Buy']]
conversion_pb['Conversion_Rate'] = conversion_pb['Buy'] / (conversion_pb['Buy'] + conversion_pb['PageView'])
conversion_pb.head()

# the conversion rate of the top 10 sale categories
conversion_10 = conversion_pb.sort_values(by='Buy', ascending=False)[:10]
conversion_10

top10_pv_categories = df_5M[df_5M['Behavior'] == 'PageView'].groupby('Category_ID').size().sort_values(ascending=False)[:10].reset_index(name="PageView Count")
top10_buy_categories = df_5M[df_5M['Behavior'] == 'Buy'].groupby('Category_ID').size().sort_values(ascending=False)[:10].reset_index(name="Buy Count")
top10_pv_categories

top10_buy_categories

# 3. Customer Segmentation
# 
# 3.1 RFM Analysis

# 1. Calculate recency
recency = df_5M[df_5M['Behavior'] == 'Buy'].groupby(by='User_ID', as_index=False)['Date'].max()
recency.columns = ['User_ID', 'LastPurshaceDate']

current_date = df_5M[df_5M['Behavior'] == 'Buy']['Date'].max()
recency['Recency'] = recency['LastPurshaceDate'].apply(lambda x: (current_date - x).days)
recency.head()

# 2. Calculate Frequency
frequency = df_5M[df_5M['Behavior'] == 'Buy'].groupby('User_ID')['Behavior'].count().reset_index()
frequency.columns = ['User_ID', 'Frequency']
frequency.head()

# 3. Create RFM table
rfm = recency.merge(frequency, on='User_ID')
rfm.drop(columns=['LastPurshaceDate'], inplace=True)
rfm.head()

# 4. Assign R, F quartile values
quantiles = rfm.quantile(q=[0.25, 0.5, 0.75])
quantiles = quantiles.to_dict()

def r_score(x):
    if x <= quantiles['Recency'][0.25]:
        return 4
    elif x <= quantiles['Recency'][0.5]:
        return 3
    elif x <= quantiles['Recency'][0.75]:
        return 2
    else:
        return 1

def f_score(x):
    if x <= quantiles['Frequency'][0.25]:
        return 1
    elif x <= quantiles['Frequency'][0.5]:
        return 2
    elif x <= quantiles['Frequency'][0.75]:
        return 3
    else:
        return 4

rfm['R_score'] = rfm['Recency'].apply(r_score)
rfm['F_score'] = rfm['Frequency'].apply(f_score)
# rfm['RFM_Class'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str)
rfm.head()

# 3. K-means Clustering-based Customer Segmentation

from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

rfm_val = rfm[['Recency', 'Frequency']]
scaler = StandardScaler()
rfm_scaled = scaler.fit(rfm_val)
rfm_scaled = scaler.fit_transform(rfm_val)
rfm_scaled

rfm_val

# determine the number of clusters using Elbow method
plt.figure(figsize=(6, 4))
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12), timings=False)
visualizer.fit(rfm_scaled)  
plt.savefig('Elbow_Plot.png')
visualizer.show()

# The optimal cluster value (K) for Elbow was found to be 3.

kmeans_scaled = KMeans(3)
kmeans_scaled.fit(rfm_scaled)
identified_clusters = kmeans_scaled.fit_predict(rfm_val)

rfm['Cluster'] = kmeans_scaled.fit_predict(rfm_scaled)
rfm.head()

# Filter the original dataset based on cluster labels
cluster_0 = clusters_scaled[clusters_scaled['cluster_pred'] == 0]
cluster_1 = clusters_scaled[clusters_scaled['cluster_pred'] == 1]
cluster_2 = clusters_scaled[clusters_scaled['cluster_pred'] == 2]

cluster_0.describe()

cluster_1.describe()

cluster_2.describe()

def categorize_customers(row):
    if row['Cluster'] == 0:
        return 'Churn Risk Customers'
    elif row['Cluster'] == 1:
        return 'Potential Customers'
    elif row['Cluster'] == 2:
        return 'High-Value Customers'

rfm['class'] = rfm.apply(categorize_customers, axis=1)

rfm.head()

customer_class = rfm.groupby('class')['User_ID'].count().reset_index()
customer_class.columns = ['Customer Class', 'Counts']
customer_class.head()

plt.figure(figsize=(4, 4))
plt.pie(customer_class['Counts'], labels=customer_class['Customer Class'], autopct='%1.1f%%', explode=[0.05]*3)
plt.title('RFM Analysis', fontsize=15)

plt.savefig('RFM Analysis.png', format='png', bbox_inches='tight')
plt.show()

# 4. Predictive Behavior Modeling

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from sklearn.metrics import accuracy_score

df_5M_time = df_5M.sort_values(by=['User_ID', 'Datetime'])
behavior_count = df_5M_time.groupby(['Product_ID', 'User_ID'])['Behavior'].size().reset_index(name='Behavior_Count')
filtered_bc = behavior_count[behavior_count['Behavior_Count'] >= 5]
df_5M_filtered = df_5M_time.merge(filtered_bc, on=['Product_ID', 'User_ID'])
df_5M_filtered.head()

sequence = df_5M_filtered.groupby(['Product_ID', 'User_ID'])['Behavior'].apply(list).reset_index(name='Sequence')
sequence.drop(columns=['Product_ID'], inplace=True)
sequence.head()

sequence_train, sequence_test = train_test_split(sequence, test_size=0.2, random_state=42)
sequence_val, sequence_test = train_test_split(sequence_test, test_size=0.5, random_state=42)

sequence_train.head()

# LSTM 
PADDING_IDX = 4
class UserBehaviorDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.token_to_idx = {'PageView': 0, 'AddToCart': 1, 'Buy': 2, 'Favorite': 3,  'Padding': PADDING_IDX}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sequence = self.dataframe.iloc[idx, 1]
        sequence_idx = [self.token_to_idx[action] for action in sequence]
        return torch.tensor(sequence_idx), torch.tensor(len(sequence_idx))

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = torch.tensor(yy)
    xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad, x_lens

train_dataset = UserBehaviorDataset(sequence_train)
val_dataset = UserBehaviorDataset(sequence_val)
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate)

class LSTMBehaviorModel(nn.Module):
    def __init__(self, num_tokens, hidden_size=128, num_layers=2):
        super(LSTMBehaviorModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(num_tokens, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_tokens)

    def forward(self, x, x_lens):
        x = self.embed(x)
        packed_x = pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        packed_out, _ = self.lstm(packed_x, (h0, c0))
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.fc(out)
        return out

# Hyper-parameters
model = LSTMBehaviorModel(num_tokens=5).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=PADDING_IDX, reduction='none').to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float('inf')
num_epochs = 5  
train_loss_list = []
val_loss_list = []
for epoch in range(num_epochs):
    model.train()
    for x_padded, x_lens in tqdm(train_loader):
        x_padded, x_lens = x_padded.to(device), x_lens.to(device)
        optimizer.zero_grad()
        output = model(x_padded, x_lens)
        loss = criterion(output.view(-1, 5), x_padded.view(-1))
        loss = loss.view(x_padded.size(0), -1).sum(dim=1).mean()
        train_loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for x_padded, x_lens in val_loader:
            x_padded, x_lens = x_padded.to(device), x_lens.to(device)
            output = model(x_padded, x_lens)
            val_loss = criterion(output.view(-1, 5), x_padded.view(-1))
            #val_loss = val_loss / x_padded.size(0)
            val_loss = val_loss.view(x_padded.size(0), -1).sum(dim=1).mean()
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch}: Avg. Validation Loss: {avg_val_loss}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')

# plot training loss curve
plt.figure(figsize=(10, 6))
plt.plot(list(range(0, len(train_loss_list))), train_loss_list, marker='o', color='b', label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss.png", dpi=600)
plt.show()

test_df = sequence_test 
test_dataset = UserBehaviorDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate)

model.eval()

all_predictions = []
all_labels = []
with torch.no_grad():
    for x_padded, x_lens in test_loader:
        x_padded, x_lens = x_padded.to(device), x_lens.to(device)
        output = model(x_padded, x_lens)
        preds = output.argmax(dim=-1)  
        for i, length in enumerate(x_lens):
            all_predictions.append(preds[i, length - 1].item())
            all_labels.append(x_padded[i, length - 1].item())

accuracy = accuracy_score(all_labels, all_predictions)
print(f"Test Accuracy: {accuracy}")

# rnn
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

class RNNBehaviorModel(nn.Module):
    def __init__(self, num_tokens, hidden_size=128, num_layers=1):
        super(RNNBehaviorModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(num_tokens, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_tokens)

    def forward(self, x, x_lens):
        x = self.embed(x)
        packed_x = pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        packed_out, _ = self.rnn(packed_x, h0)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.fc(out)
        return out


model = RNNBehaviorModel(num_tokens=5).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=PADDING_IDX, reduction='none').to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float('inf')
num_epochs = 5  

for epoch in range(num_epochs):
    model.train()
    for x_padded, x_lens in tqdm(train_loader):
        x_padded, x_lens = x_padded.to(device), x_lens.to(device)
        optimizer.zero_grad()
        output = model(x_padded, x_lens)
        loss = criterion(output.view(-1, 5), x_padded.view(-1))
        #loss = loss / x_padded.size(0)
        loss = loss.view(x_padded.size(0), -1).sum(dim=1).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for x_padded, x_lens in val_loader:
            x_padded, x_lens = x_padded.to(device), x_lens.to(device)
            output = model(x_padded, x_lens)
            val_loss = criterion(output.view(-1, 5), x_padded.view(-1))
            val_loss = val_loss.view(x_padded.size(0), -1).sum(dim=1).mean()
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch}: Avg. Validation Loss: {avg_val_loss}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')

model = RNNBehaviorModel(num_tokens=5).to(device)
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for x_padded, x_lens in test_loader:
        x_padded, x_lens = x_padded.to(device), x_lens.to(device)
        output = model(x_padded, x_lens)
        preds = output.argmax(dim=-1) 
        for i, length in enumerate(x_lens):
            all_predictions.append(preds[i, length - 1].item())
            all_labels.append(x_padded[i, length - 1].item())
            
accuracy = accuracy_score(all_labels, all_predictions)

print(f"Test Accuracy: {accuracy}")


