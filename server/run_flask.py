import sys
import os
sys.path.append('/home/sultanan/energy')
from energy_pred import GicaHackDataLoader
from flask import Flask, jsonify, request
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from pyngrok import ngrok

app = Flask(__name__)

# Load data
DATA_DIR = os.environ.get('DATA_DIR', '/home/sultanan/datasets/energy/GicaHack')  # Adjustable for Heroku later
loader = GicaHackDataLoader(DATA_DIR, verbose=True)
loader.load()
raw_df = loader.get_raw()
total_hourly = raw_df['import_diff'].groupby(raw_df['timestamp'].dt.hour).sum()

# Fit ARIMA model for forecasting
model = ARIMA(total_hourly, order=(1, 0, 0))  # Simple AR(1) model
model_fit = model.fit()

@app.route('/predict_consumption', methods=['POST'])
def predict_consumption():
    data = request.get_json()  # Expect JSON with 1,000 users' latest readings
    update_df = pd.DataFrame(data)
    
    # Calculate latest delta (assuming 'energy_import' matches your update format)
    latest_delta = update_df['energy_import'].diff().iloc[-1] if len(update_df) > 1 else 0
    if latest_delta == 0 and not update_df.empty:
        latest_delta = update_df['energy_import'].iloc[-1] - raw_df['import_diff'].iloc[-1]  # Approx from last known

    # Predict next 6 hours (challenge requires actionable forecasts)
    forecast = model_fit.forecast(steps=6)
    adjusted_forecast = forecast + (latest_delta * 4)  # Scale 15-min delta to hourly

    # Add basic peak detection (simplified for MVP)
    peak_threshold = total_hourly.mean() + total_hourly.std() * 2
    peaks = [f if f > peak_threshold else 0 for f in adjusted_forecast]
    
    return jsonify({
        'prediction': adjusted_forecast.tolist(),
        'peaks_detected': peaks,  # Indicate potential stress points
        'unit': 'Wh',
        'timestamp': pd.Timestamp.now().isoformat()
    })

if __name__ == '__main__':
    # Open ngrok tunnel
    public_url = ngrok.connect(5000)
    print(f"Ngrok tunnel URL: {public_url}")  # Share this with your team

    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)

    # Close tunnel on app exit (optional)
    ngrok.kill()