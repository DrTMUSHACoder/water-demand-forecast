from flask import Flask, render_template, request, jsonify
import sqlite3
try:
    import tensorflow as tf
except ImportError:
    tf = None
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pandas import concat
from pandas import DataFrame
import math
import random

app = Flask(__name__)
DB_NAME = 'water_demand.db'

# Redirect root to Home
@app.route('/')
def index():
    return render_template('Home_page.html')

# Load your trained model
try:
    if tf:
        model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), 'templates', 'SIH2024.keras'))
    else:
        model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load and preprocess data
try:
    csv_path = os.path.join(os.path.dirname(__file__), 'templates', 'Water Audit final 100 years data.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, header=0, index_col=0)
        values = df.values.astype('float32')
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
    else:
        raise FileNotFoundError("CSV file not found")

except Exception as e:
    print(f"Warning: Could not load data file ({e}). Using mock data structure.")
    # Create dummy data with same shape (12 columns)
    dummy_data = np.random.rand(100, 12).astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dummy_data)
    df = pd.DataFrame(dummy_data) # Mock df

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

reframed = series_to_supervised(scaled, n_in=1, n_out=1)
n_features = df.shape[1]
reframed.drop(reframed.columns[range(n_features, n_features * 2 - 1)], axis=1, inplace=True)

# Split into train and test sets
n_train_months = 36
values = reframed.values
train = values[:n_train_months, :]
test = values[n_train_months:, :]
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

def forecast_next_steps(model, last_obs, scaler, steps=1):
    forecasts = []
    current_input = last_obs

    for _ in range(steps):
        yhat = model.predict(current_input, verbose=0)
        forecasts.append(yhat[0, 0])
        current_input = np.concatenate((current_input[:, :, 1:], yhat[:, np.newaxis, :]), axis=2)

    forecasts_scaled = np.concatenate((np.zeros((len(forecasts), scaler.n_features_in_ - 1)), np.array(forecasts).reshape(-1, 1)), axis=1)
    forecasts_rescaled = scaler.inverse_transform(forecasts_scaled)[:, -1]

    return forecasts_rescaled

def mock_forecast(steps=1):
    """Generate dummy forecast data when model is missing"""
    # Generate random values around a mean similar to Consumption data (e.g., 0-10 range based on CSV)
    return [random.uniform(0.5, 10.0) for _ in range(steps)]

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

#1st page
@app.route('/Home')
def Home():
    return render_template('Home_page.html')

#2nd page
@app.route('/Expect_sol')
def Expect_sol():
    return render_template('Expect_sol.html')

#3rd page
@app.route('/Data_set', methods=['GET', 'POST'])
def Data_set():
    conn = get_db_connection()
    cursor = conn.cursor()

    if request.method == 'POST':
        # Fetch the selected district from the form
        district_1 = request.form['did']
        
        # Query to fetch data for the selected district
        cursor.execute("SELECT * FROM water_audit_dataset WHERE did = ?", (district_1,))
    else:
        # Default query to fetch all data
        cursor.execute("SELECT * FROM water_audit_dataset")
    
    data = cursor.fetchall()
    conn.close()

    return render_template('Dataset.html', data=data)

#4th Page
@app.route('/Reservoir_data', methods=['GET', 'POST'])
def Reservoir_data():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Pagination settings
    per_page = 100  # Number of rows per page
    page = request.args.get('page', 1, type=int)  # Get the current page (default to 1)

    if request.method == 'POST':
        # Fetch the selected district from the form
        district = request.form['rid']
        
        # Query to fetch data for the selected district with pagination
        offset = (page - 1) * per_page
        cursor.execute("SELECT * FROM chemmbarambakkam WHERE rid = ? LIMIT ? OFFSET ?", (district, per_page, offset))
    else:
        # Default query to fetch all data with pagination
        offset = (page - 1) * per_page
        cursor.execute("SELECT * FROM chemmbarambakkam LIMIT ? OFFSET ?", (per_page, offset))

    data = cursor.fetchall()

    # Count total rows for pagination
    cursor.execute("SELECT COUNT(*) FROM chemmbarambakkam")
    total_rows = cursor.fetchone()[0]
    total_pages = math.ceil(total_rows / per_page)

    conn.close()

    return render_template('Reservoir.html', data=data, page=page, total_pages=total_pages)

@app.route('/imdb')
def imdb():
    return render_template('imdb.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    global test_X
    try:
        years = request.form.get('years')
        months = request.form.get('months')
        
        if years is None or months is None:
            return jsonify({"error": "Both 'years' and 'months' parameters are required."})
        
        years = int(years)
        months = int(months)

        total_months_to_forecast = (years - 2024) * 12 + months
        stepz = (years - 2024) * 12
        if stepz < 0: stepz = 0 # Handle past years gracefully
        
        # Calculate result range
        start = max(0, stepz - 12)
        end = months + start
        
        # TRY-CATCH for Model Prediction (Fallback to Mock if AI fails/crashes)
        try:
            if model is None:
                raise Exception("Model not loaded")

            if len(test_X.shape) == 2:
                test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

            last_obs = test_X[-1].reshape((1, test_X.shape[1], test_X.shape[2]))
            
            next_steps = forecast_next_steps(model, last_obs, scaler, steps=max(stepz, end))
            result=next_steps[start:end]
            
            return jsonify({"forecast": result.tolist()})

        except Exception as e:
            print(f"AI Prediction Failed: {e}. Falling back to Mock Mode.")
            # Fallback to Mock Data seamlessly
            needed = end
            full_mock_data = mock_forecast(steps=needed)
            result = full_mock_data[start:end]
            return jsonify({"forecast": result, "note": f"AI Error: {str(e)}. Using Mock Data."})

    except ValueError:
        return jsonify({"error": "Please enter valid numeric values."})
    except Exception as e:
        return jsonify({"error": str(e)})
    
#4th page
@app.route('/ML_Algorithms')
def ML_Algorithms():
    return render_template('ML-Algorithms.html')

#6th page
@app.route('/Res_con')
def Res_con():
    return render_template('Res-con.html')

@app.route('/Dashboard')
def Dashboard():
    return render_template('Dashboard.html')

#7th page
@app.route('/Water_demo_video')
def Water_demo_video():
    return render_template('Demo_video.html')

if __name__ == '__main__':
    app.run(debug=True)