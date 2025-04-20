
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set Streamlit page config
st.set_page_config(page_title="Production Forecasting Capstone", layout="wide")

# Title
st.title('Production Forecasting Capstone App')

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('2021-2024.csv')
    return data

data = load_data()
st.write("### Raw Data", data.head())

# Feature Engineering
def create_features(df):
    df['MA_3'] = df['Production ex KPC (kt)'].rolling(window=3).mean()
    df['MA_7'] = df['Production ex KPC (kt)'].rolling(window=7).mean()
    df['MA_14'] = df['Production ex KPC (kt)'].rolling(window=14).mean()
    df['EWMA_7'] = df['Production ex KPC (kt)'].ewm(span=7, adjust=False).mean()
    df['EWMA_14'] = df['Production ex KPC (kt)'].ewm(span=14, adjust=False).mean()
    for lag in range(1, 8):
        df[f'lag_{lag}'] = df['Production ex KPC (kt)'].shift(lag)
    df['dayofweek'] = pd.to_datetime(df['date']).dt.dayofweek
    df['month'] = pd.to_datetime(df['date']).dt.month
    return df

data = create_features(data)
data = data.dropna()

# Features and Target
features = [
    'Fuel Gas - Total Produced (Mscm)', 'Raw Gas to OGP (Mscm)', 'Gas Injection (Mscm)',
    'MA_3', 'MA_7', 'MA_14', 'EWMA_7', 'EWMA_14',
    'dayofweek', 'month',
    'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7'
]
target = 'Production ex KPC (kt)'

X = data[features]
y = data[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader('Model Evaluation')
st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")

# Plot Actual vs Predicted
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs Predicted')
st.pyplot(fig)
