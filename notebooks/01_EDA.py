# %% [markdown]
# # 🎯 Problem Understanding
#
# **Business Problem**:
# Banks use telemarketing to sell term deposits to their clients. However, running these campaigns is expensive and often yields low conversion rates. The bank wants to reduce marketing costs and improve efficiency by targeting only those clients who are most likely to subscribe to a term deposit.
#
# **Objective**:
# Build a Machine Learning model that predicts whether a client will subscribe to a term deposit based on their demographic information and past interactions with the bank.
#
# **Success Metrics**:
# - **Accuracy**: Overall correct predictions (may be misleading if data is imbalanced).
# - **Precision**: Out of all clients predicted as 'Yes' for subscription, how many actually subscribed? (Minimizes false positives - wasting time on people who won't buy).
# - **Recall**: Out of all actual subscribers, how many did we correctly identify? (Minimizes false negatives - missing out on potential customers).
# - **F1-Score**: Harmonic mean of Precision and Recall.
# - **ROC-AUC**: Shows the model's ability to distinguish between classes (Subscribers vs Non-subscribers).
#
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plot style for premium Look
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#1e1e1e", "figure.facecolor":"#121212", "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})

# %% [markdown]
# # 🧠 Data Understanding
# Let's load the data from our `data/` directory.

# %%
# Load dataset
data_path = '../data/data.csv'
try:
    df = pd.read_csv(data_path, sep=';') # Bank marketing is often separated by ';'
    if len(df.columns) <= 1:
         df = pd.read_csv(data_path, sep=',') # fallback if standard comma
except Exception as e:
    print(f"Error loading file: {e}. Attempting standard comma separation.")
    df = pd.read_csv(data_path, sep=',')

print("Dataset Loaded Successfully!")
print("-" * 50)
print(f"Shape of the dataset: {df.shape[0]} rows and {df.shape[1]} columns")
print("-" * 50)
print("Data Types:\n", df.dtypes)
print("-" * 50)
print("Missing Values per column:\n", df.isnull().sum())
print("-" * 50)
print("Number of Unique Values per column:\n", df.nunique())

# %% [markdown]
# ### Exploring Specific Columns
# - **pdays**: number of days that passed by after the client was last contacted from a previous campaign (999 means client was not previously contacted)
# - **previous**: number of contacts performed before this campaign and for this client
# - **poutcome**: outcome of the previous marketing campaign (failure, nonexistent, success)
# - **y**: has the client subscribed a term deposit? (target variable)

# %%
print("\nTarget Variable Distribution 'y':")
if 'y' in df.columns:
    print(df['y'].value_counts(normalize=True) * 100)
else:
    print("Warning: Target column 'y' not found. It might be named differently.")

# %% [markdown]
# The target variable `y` is highly imbalanced in typical bank datasets (e.g. ~88% No, ~12% Yes). Class imbalance will need to be handled during modeling.

# %% [markdown]
# # 📊 Exploratory Data Analysis (EDA)

# %%
# 1. Univariate Analysis (Categorical Variables)
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
if 'y' in cat_cols: cat_cols.remove('y')

if len(cat_cols) > 0:
    fig, axes = plt.subplots(len(cat_cols)//2 + len(cat_cols)%2, 2, figsize=(15, 20))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        sns.countplot(y=col, data=df, ax=axes[i], palette='viridis', order=df[col].value_counts().index)
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_ylabel('')

    # Remove any empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig('univariate_categorical.png', dpi=300, bbox_inches='tight')
    print("Saved univariate_categorical.png")

# %% [markdown]
# **Insight**: Most clients are typically in 'admin', 'blue-collar' or 'technician' roles. Most are married and have a university degree.

# %%
# 2. Bivariate Analysis (Target vs Categorical) - if 'y' exists
if len(cat_cols) > 0 and 'y' in df.columns:
    fig, axes = plt.subplots(len(cat_cols)//2 + len(cat_cols)%2, 2, figsize=(15, 20))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        sns.countplot(y=col, hue='y', data=df, ax=axes[i], palette='magma', order=df[col].value_counts().index)
        axes[i].set_title(f'{col} vs Subscription (y)')
        axes[i].set_ylabel('')

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig('bivariate_categorical.png', dpi=300, bbox_inches='tight')
    print("Saved bivariate_categorical.png")

# %% [markdown]
# **Insight**: 
# - Students and retired people have a slightly higher proportional conversion rate relative to their size.
# - The 'success' outcome from the previous campaign (`poutcome`) is a very strong indicator of a subscribe 'yes' in the current campaign.

# %%
# 3. Correlation Analysis (Numerical Variables)
num_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(num_cols) > 0:
    plt.figure(figsize=(10, 8))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved correlation_matrix.png")

# %% [markdown]
# **Insight**: 
# - `pdays` and `previous` might have a correlation as both relate to previous contact.
# - We might expect multicollinearity if economic indicators (like euribor3m) are present.

# %%
print("\nEDA Completed Successfully! Visualizations exported as PNG files.")
