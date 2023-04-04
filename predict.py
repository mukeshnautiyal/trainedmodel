import pandas as pd
from keras.models import load_model


def predict_earning():
    model = load_model('trained_model10.h5')

    X = pd.read_csv("data.csv").values
    prediction = model.predict(X)

    # Grab just the first element of the first prediction (since we only have one)
    print(f'prediction - {prediction}')
    prediction = prediction[0][0]

    # Re-scale the data from the 0-to-1 range back
    # These constants are from when the data was originally scaled down to the 0-to-1 range
    prediction = prediction + 0.1159
    prediction = prediction / 0.0000036968

    print("Earnings Prediction for Proposed Product - {}".format(prediction))


if __name__ == '__main__':
    predict_earning()

