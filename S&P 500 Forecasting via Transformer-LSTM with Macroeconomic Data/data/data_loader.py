import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_data_sliding(
        file_path,
        time_col,
        feature_cols,
        target_cols,
        window_size=12,
        horizon=1,
        train_ratio=0.7,
        val_ratio=0.15,
        normalization=None  # ('minmax','standard' or None)
):

    #  Read data
    df = pd.read_excel(file_path)
    pd.set_option('display.max_columns', None)
    print(df.head())
    if time_col in df.columns:
        df.sort_values(by=time_col, inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        print(f"[Warning] time_col={time_col} not found in the table, skipping sorting.")

    # Time column for figure plotting
    if time_col in df.columns:
        time_data = df[time_col].values
    else:
        time_data = np.arange(len(df))

    # Extract feature and target data
    feature_data = df[feature_cols].values  # shape = (N, F)
    target_data = df[target_cols].values  # shape = (N, T)  (T=1 or more)

    # Normalization
    scaler_X = None
    scaler_y = None

    if normalization == 'minmax':
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        feature_data = scaler_X.fit_transform(feature_data)
        target_data = scaler_y.fit_transform(target_data)
    elif normalization == 'standard':
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        feature_data = scaler_X.fit_transform(feature_data)
        target_data = scaler_y.fit_transform(target_data)

    # Sliding windows
    X_list = []
    Y_list = []
    time_list = []

    max_index = len(feature_data) - window_size - horizon + 1

    for i in range(max_index):
        x_slice = feature_data[i: i + window_size, :]
        y_slice = target_data[i + window_size: i + window_size + horizon, :]

        X_list.append(x_slice)
        Y_list.append(y_slice.reshape(horizon, -1))

        time_list.append(time_data[i + window_size + horizon - 1])

    X_array = np.array(X_list)  # shape = (number of sample, window_size, F)
    Y_array = np.array(Y_list)  # shape = (number of sample, horizon, T)

    if Y_array.shape[-1] == 1:
        Y_array = Y_array.squeeze(-1)  # => (number of sample, horizon)

    time_array = np.array(time_list)  # shape = (number of sample,)

    # Split into train/validation/test sets
    num_samples = X_array.shape[0]
    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))

    X_train = X_array[:train_end]
    y_train = Y_array[:train_end]
    train_time = time_array[:train_end]

    X_val = X_array[train_end:val_end]
    y_val = Y_array[train_end:val_end]
    val_time = time_array[train_end:val_end]

    X_test = X_array[val_end:]
    y_test = Y_array[val_end:]
    test_time = time_array[val_end:]

    return (X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            scaler_X, scaler_y,
            val_time, test_time)


# for testing
if __name__ == "__main__":
    file_path = 'enhanced raw data.xlsx'
    time_col = 'Date'
    feature_cols = [
        'EFFR', 'CPI', 'GEU', 'HY Index', 'HPI',
        'PPI', '3m T-Yield', '10y T-Yield', 'UR', 'DTI',
        'Exports', 'Imports', 'GDP', 'Crude Oil', 'BTC', 'Party',
        'Gold', 'year', 'year_offset', 'month', 'sin_month', 'cos_month'
    ]
    target_cols = ['S&P 500 Price']

    # Updated function call with validation set
    (X_train, y_train,
     X_val, y_val,
     X_test, y_test,
     scaler_X, scaler_y,
     val_time, test_time) = load_data_sliding(
        file_path=file_path,
        time_col=time_col,
        feature_cols=feature_cols,
        target_cols=target_cols,
        window_size=12,
        horizon=1,
        train_ratio=0.7,  # Adjusted for validation split
        val_ratio=0.15,  # Added validation ratio
        normalization='minmax'
    )

    # Updated print statements
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)  # New
    print("y_val shape:", y_val.shape)  # New
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    if scaler_X:
        print("Scaler_X:", scaler_X)
    if scaler_y:
        print("Scaler_Y:", scaler_y)

    print("val_time shape:", val_time.shape)  # New
    print("test_time shape:", test_time.shape)

