import pandas as pd
import numpy as np
from keras.src.optimizers import Adam, RMSprop
from taipy import Gui
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model_performance(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return rmse, r2, mae
    # print(f"MAE: {mae}")
    # print(f"MSE: {mse}")
    # print(f"RMSE: {rmse}")
    # print(f"R²: {r2:.2f}")

data_set = pd.DataFrame(pd.read_csv("AAPL.csv", parse_dates=True)).set_index("Date")

# Bifurcating the data set into train, test and val
def train_test_val(data):
    train_size, val_size = int(len(data) * 0.7), int(len(data) * 0.15)
    train, test, val = data.iloc[:train_size], data.iloc[train_size:train_size + val_size], data.iloc[train_size + val_size:]


    x_train = train.drop(columns=["Close"])
    y_train = train["Close"]

    x_test = test.drop(columns=["Close"])
    y_test = test["Close"]

    x_val = val.drop(columns=["Close"])
    y_val = val["Close"]
    return x_train, x_test, x_val, y_test, y_train, y_val

x_train, x_test, x_val, y_test, y_train, y_val = train_test_val(data_set)
print(f"Shape of x_train : {x_train.shape} and x_test :{x_test.shape} and y_train : {y_train.shape} and y_test : {y_test.shape}")

#Preprocessing of the dataframe:
from sklearn.preprocessing import MinMaxScaler
# Scale features
scaler_x = MinMaxScaler()
x_train_scaled = scaler_x.fit_transform(x_train)
x_test_scaled = scaler_x.transform(x_test)
x_val_scaled = scaler_x.transform(x_val)
# Scale targets
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1,1))
print(f"Shape of x_train :{x_train_scaled.shape} and x_test :{x_test_scaled.shape} and x_val : {x_val_scaled.shape} and y_train : {y_train_scaled.shape} and y_test : {y_test_scaled.shape} y_val: {y_val_scaled.shape}")

def add_noise(data, noise_factor=0.01):
    noise = np.random.normal(loc=0, scale=noise_factor, size=data.shape)
    return data + noise

# Augment the training data
x_train_augmented = add_noise(x_train_scaled)
y_train_augmented = y_train  # Keep the same target values

# Combine original and augmented data
x_train_combined = np.vstack([x_train_scaled, x_train_augmented])
y_train_combined = np.vstack([y_train, y_train_augmented])

x_val_augmented = add_noise(x_val_scaled, noise_factor=0.01)
y_val_augmented = add_noise(y_val_scaled, noise_factor=0.01)
x_test_augmented = add_noise(x_test_scaled, noise_factor=0.01)
y_test_augmented = add_noise(y_test_scaled, noise_factor=0.01)

print(f"Shape of x_train :{x_train_combined.shape}  and y_train : {y_train_combined.shape} ")
y_train_combined=y_train_combined.reshape(-1,1)
print(f"Shape of y_train : {y_train_combined.shape}")
y_train_combined = scaler_y.transform(y_train_combined)

from keras._tf_keras.keras.preprocessing.sequence import TimeseriesGenerator
n_input = 50
n_features = x_train_combined.shape[1]
train_generator = TimeseriesGenerator(x_train_combined, y_train_combined, length=n_input, batch_size=10)
test_generator = TimeseriesGenerator(x_test_scaled, y_test_scaled, length=n_input, batch_size=10)
val_generator = TimeseriesGenerator(x_val_augmented, y_val_augmented, length=n_input, batch_size=10)


from keras.api.models import Sequential
from keras.api.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.layers import Bidirectional


import optuna
from keras.api.models import load_model

# model = Sequential()
# model.add(Bidirectional(LSTM(50, activation="tanh", return_sequences=False, input_shape=(n_input, n_features))))
# # model.add(LSTM(12, activation="tanh", return_sequences=False, recurrent_dropout=0.2))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss="mean_squared_error")
# early_stopping = EarlyStopping(monitor="loss", patience=2)
# history = model.fit(train_generator, epochs=10, batch_size=10, callbacks=[early_stopping])
# model.save("Raw_Model.keras")

model=load_model("Raw_Model.keras")

predictions = model.predict(test_generator)
predicted_prices = scaler_y.inverse_transform(predictions)
rmse, r2, mae = evaluate_model_performance(y_test[50:], predicted_prices)



# Prepare data for the chart
def prepare_combined_chart_data(y_true, y_pred):
    data = pd.DataFrame({
        "Index": range(len(y_true)),
        "Actual": y_true.flatten(),
        "Predicted": y_pred.flatten(),
    })
    return data

# Prepare combined data for the chart
combined_chart_data = prepare_combined_chart_data(y_test[50:].values, predicted_prices)


page = """
<|text-center|
<h1>Stock Closing Price Prediction</h1>
>
<h5>Here I learned how to use LSTM model which is an advanced technique of RNN for time-series analysis. LSTM stands for long-short term memory model. Since stock analysis is a time-series analysis we will be using this model to predict closing price.</h5>

<h5> It is a kaggle competition and the code is pushed on GitHub repository.</h5>

<|{combined_chart_data}|chart|type=line|x=Index|y[1]=Actual|y[2]=Predicted|title=Actual vs Predicted Closing Prices|>

<h5> The following are the metrics of the model, this one is optimized or fine-tuned by optuna : </h5>

<|layout|columns = 1 1 1 1|
# MAE : <|metric|value={mae:.2f}|>
# RMSE : <|metric|value={rmse:.2f}|>
# R² : <|metric|value={r2:.2f}|>
|>
"""

app = Gui(page)
if __name__ == "__main__":
    app.run(title="Bi-LSTM", host="0.0.0.0", port=5001)

