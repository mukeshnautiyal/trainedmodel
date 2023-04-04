import pandas as pd
from keras.models import Sequential
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler


def preprocess_data():
    # Load training data set from CSV file
    training_data_df = pd.read_csv("data.csv")
    print(training_data_df)
    # Load testing data set from CSV file
    test_data_df = pd.read_csv("data_test.csv")

    # Data needs to be scaled to a small range like 0 to 1 for the neural
    # network to work well.
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Scale both the training inputs and outputs
    scaled_training = scaler.fit_transform(training_data_df)
    print(scaled_training)
    scaled_testing = scaler.transform(test_data_df)

    # Create new pandas DataFrame objects from the scaled data
    scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
    scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)

    # Save scaled data dataframes to new CSV files
    scaled_training_df.to_csv("data_training_scaled.csv", index=False)
    scaled_testing_df.to_csv("data_testing_scaled.csv", index=False)


def training():
    training_data_df = pd.read_csv("data_training_scaled.csv")

    X = training_data_df.drop('class attribute', axis=1).values
    print(X,"xxx")
    Y = training_data_df[['class attribute']].values
    print(Y,"hyyyy")

    # Define the model
    model = Sequential()
    model.add(Dense(50, input_dim=10, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss="mean_squared_error", optimizer="adam")

    # Train the model
    
    model.fit(X, Y, epochs=100, shuffle=True, verbose=2)

    # Load the separate test data set
    test_data_df = pd.read_csv("sales_data_testing_scaled.csv")

    X_test = test_data_df.drop('class attribute', axis=1).values
    Y_test = test_data_df[['class attribute']].values

    test_error_rate = model.evaluate(X_test, Y_test, verbose=5)
    print("The mean squared error for the test data set is: {}".format(test_error_rate))

    # Save the model to disk
    model.save("trained_model10.h5")


if __name__ == '__main__':
    preprocess_data()
    training()
