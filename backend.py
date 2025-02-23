from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Define function to create sequences
def create_sequences(data, time_steps=10):
    X_seq = []
    for i in range(len(data) - time_steps):
        X_seq.append(data[i:i + time_steps])  # Create rolling sequences
    return np.array(X_seq)

@app.route('/predict', methods=['GET'])

#@app.route('/')
#def hello_world():
#    return 'Hello, World!'

def predict():
    model = tf.keras.models.load_model("C:/Users/pokkd/Downloads/saved_model.keras")
    df = pd.read_excel("C:/Users/pokkd/Downloads/ur3+cobotops/dataset_02052023.xlsx", nrows=1000)
    df = df.dropna()
    df['Timestamp'] = df['Timestamp'].str.strip('"')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['timestamp'] = df['Timestamp'].astype('int64')
    df['robot_fail'] = df['grip_lost'] | (df['Robot_ProtectiveStop'] == 1)
    df = df.drop(columns = ['Timestamp', 'Num', 'Robot_ProtectiveStop', 'grip_lost', 'cycle ', 'robot_fail'])
    #num_failures = df[df['robot_fail'] == True].shape[0]
    #num_normals = df[df['robot_fail'] == False].shape[0]
    #Maybe delete this
    #df = df.drop(columns = ['Timestamp'])
    #failure_ratio = num_failures / (num_failures + num_normals)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_X = scaler.fit_transform(df)
    # Apply function
    X_seq = create_sequences(scaled_X, 10)  # Ensure shape is (samples, 10, 21)
    #data = request.get_json()
    #input_data = np.array(data['input'])

    predictions = model.predict(X_seq)
    X_seq = X_seq.ravel()
    prediction_error = np.mean(np.abs(predictions - X_seq), axis=1)
    threshold = np.percentile(prediction_error, 95) # This value can be adjusted based on your data and model performance
    anomalies = prediction_error > threshold
    anomaly_indices = np.where(anomalies)[0]

    anomaly_errors = prediction_error[anomaly_indices]

    return jsonify({
        'Predicted Failures': anomaly_indices.tolist(),  # List of indices where anomalies were detected
        'prediction_error': anomaly_errors.tolist()  # Prediction errors for further analysis
    })

if __name__ == "__main__":
    app.run(debug=True)
