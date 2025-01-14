import pandas as pd
import numpy as np
from taipy import Gui
import os

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

#Preprocessing of the dataframe:
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_val_scaled = scaler.transform(x_val)


y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
y_val = pd.DataFrame(y_val)

scaler = MinMaxScaler()
scaler.fit(y_train)
y_train_scaled = scaler.transform(y_train)
y_test_scaled = scaler.transform(y_test)
y_val_scaled = scaler.transform(y_val)



from keras._tf_keras.keras.preprocessing.sequence import TimeseriesGenerator
n_input = 10
n_features = x_train_scaled.shape[1]
train_generator = TimeseriesGenerator(x_train_scaled, y_train_scaled, length=n_input, batch_size=16)
test_generator = TimeseriesGenerator(x_test_scaled, y_test_scaled, length=n_input, batch_size=16)
val_generator = TimeseriesGenerator(x_val_scaled, y_val_scaled, length=n_input, batch_size=16)


from keras.api.models import Sequential
from keras.api.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import optuna
# def objective(trial):
#     # Hyperparameter search space
#     n_units = trial.suggest_int("n_units", 16, 128)
#     dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
#     learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
#     batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
#     optimizer = trial.suggest_categorical("optimizer", ["adam", "rmsprop", "sgd"])
#     time_steps = trial.suggest_int("time_steps", 10, 60)
#
#     # Build and train LSTM model using these hyperparameters
#     model = Sequential()
#     model.add(LSTM(n_units, activation="relu", input_shape=(n_input, n_features)))
#     model.add(Dropout(rate=dropout_rate))
#     model.add(Dense(1))
#     model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
#     early_stopping = EarlyStopping(monitor="val_loss", patience=5)
#     history = model.fit(
#         train_generator,
#         validation_data=val_generator,
#         epochs=10,
#         batch_size=batch_size,
#         callbacks=[early_stopping]
#     )
#     model.save('my_model.keras')
#     val_predictions = model.predict(val_generator)
#     val_predictions = scaler.inverse_transform(val_predictions)
#     y_val_trimmed = scaler.inverse_transform(y_val_scaled[10:])
#     rmse = np.sqrt(mean_squared_error(y_val_trimmed, val_predictions))
#     return rmse
#
# def run_optuna_optimization():
#     study = optuna.create_study(direction="minimize")  # Minimize RMSE
#     study.optimize(objective, n_trials=25)  # Run 5 trials
#
#     # Print the best parameters and best score
#     print("Best RMSE: ", study.best_value)
#     print("Best Hyperparameters: ", study.best_params)
#
#     return study
#
# # Run the study
# study = run_optuna_optimization()


model = Sequential()
model.add(LSTM(50, activation="relu", input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss="mean_squared_error")
early_stopping = EarlyStopping(monitor="val_loss", patience=5)
history = model.fit(train_generator, epochs=7, batch_size=16, callbacks=[early_stopping])

from keras.api.models import load_model
fine_model = load_model("my_model.keras")

predictions = model.predict(test_generator)
predicted_prices = scaler.inverse_transform(predictions)

fine_predictions = fine_model.predict(test_generator)
fine_predicted_prices = scaler.inverse_transform(fine_predictions)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model_performance(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return rmse, mape, r2, mae
    # print(f"MAE: {mae}")
    # print(f"MSE: {mse}")
    # print(f"RMSE: {rmse}")
    # print(f"MAPE: {mape:.2f}%")
    # print(f"R²: {r2:.2f}")

rmse,mape,r2, mae = evaluate_model_performance(y_test[10:], predicted_prices)
f_rmse, f_mape, f_r2, f_mae = evaluate_model_performance(y_test[10:], fine_predicted_prices)

# Prepare data for the chart
def prepare_combined_chart_data(y_true, y_pred):
    data = pd.DataFrame({
        "Index": range(len(y_true)),
        "Actual": y_true.flatten(),
        "Predicted": y_pred.flatten(),
    })
    return data

# Prepare combined data for the chart
combined_chart_data = prepare_combined_chart_data(y_test[10:].values, predicted_prices)
fine_combined_chart_data = prepare_combined_chart_data(y_test[10:].values, fine_predicted_prices)


page = """
<|text-center|
<h1>Stock Closing Price Prediction</h1>
>
<h5>Here I learned how to use LSTM model which is an advanced technique of RNN for time-series analysis. LSTM stands for long-short term memory model. Since stock analysis is a time-series analysis we will be using this model to predict closing price.</h5>

<h5> It is a kaggle competition and the code is pushed on GitHub repository.</h5>

<|{combined_chart_data}|chart|type=line|x=Index|y[1]=Actual|y[2]=Predicted|title=Actual vs Predicted Closing Prices|>

<h5> The following are the metrics of the model, this one is not optimized or fine-tuned by optuna : </h5>

<|layout|columns = 1 1 1 1|
# MAE : <|metric|value={mae:.2f}|>
# RMSE : <|metric|value={rmse:.2f}|>
# MAPE : <|metric|value={mape:.2f}%|>
# R² : <|metric|value={r2:.2f}|>
|>

<h3> Performance of Optuna fine tuned model : </h3>

<|{fine_combined_chart_data}|chart|type=line|x=Index|y[1]=Actual|y[2]=Predicted|title=Actual vs Predicted Closing Prices|>

<h5> The following are the metrics of the model, this one is optimized or fine-tuned by optuna : </h5>

<|layout|columns = 1 1 1 1|
# MAE : <|metric|value={f_mae:.2f}|>
# RMSE : <|metric|value={f_rmse:.2f}|>
# MAPE : <|metric|value={f_mape:.2f}%|>
# R² : <|metric|value={f_r2:.2f}|>
|>

"""

if __name__== "__main__":
    app = Gui(page)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, use_reloader=True, debug=True)


