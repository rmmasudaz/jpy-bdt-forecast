
import tensorflow as tf
import numpy as np
import pandas as pd
from src.train_lstm import build_model, make_windows, train_lstm_ensemble

def test_lstm_model_input_shape():
    """Test that the LSTM model accepts the correct input shape."""
    
    # Load and preprocess test data
    df = pd.read_csv('data/jpy_bdt_daily.csv', parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    rate = df["jpy_to_bdt"].to_numpy(np.float64)
    n = len(rate)
    
    # Compute log-returns of all three series
    r_jb = np.zeros(n); r_jb[1:] = np.diff(np.log(rate))
    r_uj = np.zeros(n); r_uj[1:] = np.diff(np.log(df["usd_to_jpy"].to_numpy(np.float64)))
    r_ub = np.zeros(n); r_ub[1:] = np.diff(np.log(df["usd_to_bdt"].to_numpy(np.float64)))
    
    # Create features
    feat = np.column_stack([r_jb, np.abs(r_jb), r_uj, r_ub])
    
    # Split data
    split = int(n * 0.8)
    
    # Create windows
    seq_len = 30
    X, y = make_windows(feat, seq_len)
    
    # Verify input shape (samples, time steps, features)
    assert len(X.shape) == 3, "Input should be 3-dimensional (samples, time steps, features)"
    assert X.shape[1] == seq_len, f"Time steps should be {seq_len}"
    assert X.shape[2] == 4, "Features should be 4"
    
    # Create model and test prediction
    model = build_model(seq_len, feat.shape[1])
    prediction = model.predict(X)
    
    # Verify prediction shape
    assert len(prediction.shape) == 2, "Prediction should be 2-dimensional (samples, value)"
    assert prediction.shape[0] == X.shape[0], "Number of predictions should match number of samples"
    assert prediction.shape[1] == 1, "Prediction should have 1 feature"

def test_trained_model_load():
    """Test that the trained LSTM model can be loaded and used for prediction."""
    
    try:
        model = tf.keras.models.load_model('models/lstm_model.keras')
        
        # Test with dummy data that has the correct shape (samples, 30, 4)
        dummy_input = np.random.randn(1, 30, 4)
        prediction = model.predict(dummy_input)
        
        assert prediction.shape == (1, 1), "Prediction should be (1, 1) for 1 sample"
        
        print("✅ Trained model loaded successfully and produces valid predictions")
        
    except FileNotFoundError:
        print("⚠️  Trained model file not found. Please run the training script first.")
        
    except Exception as e:
        print(f"❌ Error loading trained model: {e}")
        raise

if __name__ == "__main__":
    print("Running LSTM model tests...")
    try:
        test_lstm_model_input_shape()
        print("✅ LSTM model input shape test passed")
        
        test_trained_model_load()
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
