

from flask import Flask, request, jsonify
import pandas as pd
from utils import *  # Make sure utils.py is properly formatted and all necessary functions are defined
import logging
from tensorflow.keras.models import load_model
import joblib
model_name = 'model/buoy_LSTM_6hrlead_16tsinput.h5'


# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

app = Flask(__name__)

@app.route('/51205_6hr', methods=['GET', 'POST'])
def predict():
    try:
        logger.info("Starting prediction process...")

        # Fetch raw buoy data
        logger.info("Fetching buoy data...")
        dfs = get_buoys(buoy_ids)
        logger.info(f'\n\nlen(dfs) = {len(dfs)}\n\n')
        # print(f'dfs shape: {dfs.shape}')
        # Begin cleaning data
        logger.info("Preparing data...")
        prepped_dfs = prep_dfs(dfs, cols_to_keep)
        # print(f'prepped_dfs shape: {prepped_dfs.shape}')
        
        # Merge and clean data
        logger.info("Merging data...")
        dfx = get_merged(prepped_dfs)
        logger.info(f'\n\dfx shape (post merge): {dfx.shape}\n\n')

        # Get time of prediction
        logger.info("Getting time of prediction...")
        time_of_y = get_future_time(dfx, lead_time)
        
        # Convert columns to float
        logger.info("Converting columns to float...")
        dfx = convert_columns_to_float(dfx)

        # Load scalers from file
        logger.info("Loading scalers...")
        Xscaler, yscaler = load_scalers()

        # Align columns with the scaler
        dfx.columns = scaler_compatible_columns   
        logger.info('converting feet to meters for prediction...')
        dfx = convert_ft_to_m(dfx)  # Convert feet to meters`
        # Scale numeric, encode categories
        logger.info("Processing data...")
        dfx_processed = encodeNscale(dfx, Xscaler)

        # Ensure all dummies are present for proper array shape
        dfx_processed_aligned = align_dataframes(dfx_processed, final_X_columns)

        # Smoothing data for better prediction
        logger.info("Smoothing data...")
        dfx_processed__aligned_smoothed = smooth_dataframe(dfx_processed_aligned)
        logger.info(f'Shape of dfx is {dfx_processed__aligned_smoothed.shape}')

        # Create X array
        logger.info("Creating input array...")
        X = get_data_contiguous_X_only(dfx_processed__aligned_smoothed, n_timesteps, lead_time=lead_time, num_outputs=num_outputs, timestep_size=3600)

        # Load model
        logger.info("Loading model...")
        model = load_model(model_name)
        # model = joblib.load(model_name)

        # Get raw prediction
        logger.info("Making prediction...")
        logger.info(f'X.shape = {X.shape}')
        y_scaled = model.predict(X)

        # Scale prediction
        y = yscaler.inverse_transform(y_scaled)
        logger.info(f'converting meters back to feet for output...')
        yhat = convert_m_to_ft(y.ravel())  # Flatten the array and convert meters to feet
        logger.info(f'Prediction in meters: {yhat}, of type {type(yhat)}')
        # yhat_serializable = yhat.tolist()
        prediction_time_str = datetime_to_readable_string(time_of_y)

        # Format and return the response
        return jsonify({
            'prediction': yhat,
            'prediction_time': prediction_time_str
        })

    # except Exception as e:
    #     logger.error(f'Encountered an error: {e}', exc_info=True)
    #     return jsonify({'error': str(e)}), 500
    except ValueError as ve:
        logger.error(f'Date-Time Parsing Error: {ve}')
        return jsonify({'error': 'Date-Time Parsing Error', 'details': str(ve)}), 500
    except Exception as e:
        logger.error(f'General Error: {e}', exc_info=True)
        return jsonify({'error': 'General Error', 'details': str(e)}), 500
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
