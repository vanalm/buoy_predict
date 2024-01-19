from flask import Flask, request, jsonifyimport pandas as pd# Import other necessary libraries and modulesfrom utils import *app = Flask(__name__)model = load_model('path/to/model.sav')@app.route('/predict', methods=['GET', 'POST'])def predict():    #fetch raw buoy data    dfs = get_buoys(buoy_ids)        #begin cleaning data    prepped_dfs = prep_dfs(dfs, cols_to_keep)        #merge and clean data    dfx = get_merged(prepped_dfs)    #get time of prediction    time_of_y = get_future_time(dfx, lead_time)        dfx = convert_columns_to_float(dfx)    #load scalers from file    Xscaler, yscaler = load_scalers()    #make columns align with the scaler    dfx.columns = scaler_compatible_columns       #scale numberic, enocode cats         dfx_processed = encodeNscale(dfx, Xscaler)    #making sure that all dummies are present for poper array shape    dfx_processed_aligned = align_dataframes(dfx_processed, final_X_columns)    #smoothing data for better prediction (as verified in previous experiments)    dfx_processed__aligned_smoothed = smooth_dataframe(dfx_processed_aligned)    #create X array    X = get_data_contiguous_X_only(dfx_processed__aligned_smoothed, n_timesteps, lead_time=lead_time, num_outputs = num_outputs, timestep_size=3600)    #load model    model = load_model(model_name)    #get raw prediction    y_scaled = model.predict(X)    #scale prediction    y = yscaler.inverse_transform(y_scaled)    # Format and return the response    return jsonify({'prediction': y.tolist()})if __name__ == "__main__":    app.run(host='0.0.0.0', port=80)