"""
Train Model Script
This script trains a Linear Regression model to predict apartment rent.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os

def load_data(filepath):
    """
    Load apartment data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded data
    """
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)

    # Read CSV file
    df = pd.read_csv(filepath)
    
    print(f"✅ Loaded {len(df)} apartments from {filepath}")
    print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")

    # Show first few rows
    print(f"\nFirst 3 apartments:")
    print(df.head(3))

    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("\n✅ No missing values found!")
    else:
        print(f"\n⚠️ Warning: Missing values detected:")
        print(missing[missing > 0])

    return df

def preprocess_data(df):
    """
    Preprocess the data: encode categorical variables.
    
    Args:
        df (pd.DataFrame): Raw data
    
    Returns:
        tuple: (processed_df, encoder)
    """
    print("\n" + "="*60)
    print("STEP 2: PREPROCESSING DATA")
    print("="*60)

    # Make a copy to avoid modifying original
    df_processed = df.copy()

    # Encode locality (text → numbers)
    print("\nEncoding 'locality' column...")
    print("Before encoding (sample):")
    print(df['locality'].head())

    # Create encoder
    le = LabelEncoder()
    df_processed['locality_encoded'] = le.fit_transform(df_processed['locality'])

    print("\nAfter encoding:")
    print(df_processed[['locality', 'locality_encoded']].head())

    # Show mapping
    print("\nLocality → Number mapping:")
    for i, locality in enumerate(le.classes_):
        print(f"  {locality:20s} → {i}")

    # Save encoder for later use (in web app)
    os.makedirs('models', exist_ok=True)
    with open('models/locality_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("\n✅ Locality encoder saved to: models/locality_encoder.pkl")
    
    return df_processed, le

def train_model(df):
    """
    Train the Linear Regression model.
    
    Args:
        df (pd.DataFrame): Preprocessed data
    
    Returns:
        tuple: (model, X_test, y_test, predictions)
    """
    print("\n" + "="*60)
    print("STEP 3: TRAINING MODEL")
    print("="*60)

    feature_columns = ['sq_ft', 'bhk', 'floor', 'locality_encoded', 
                       'furnished', 'parking', 'age_years']

    X = df[feature_columns]
    y = df['rent']
    
    print(f"\nFeatures (X): {feature_columns}")
    print(f"Target (y): rent")
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}") 

    # Split into training and testing sets
    print("\n" + "-"*60)
    print("Splitting data: 80% train, 20% test")
    print("-"*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,      # 20% for testing
        random_state=42     # For reproducibility
    )
    
    print(f"Training set: {len(X_train)} apartments")
    print(f"Testing set:  {len(X_test)} apartments")

    # Create and train the model
    print("\n" + "-"*60)
    print("Training Linear Regression model...")
    print("-"*60)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("✅ Model training complete!")

    # Make predictions
    print("\n" + "-"*60)
    print("Making predictions...")
    print("-"*60)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Evaluate model
    print("\n" + "="*60)
    print("STEP 4: MODEL EVALUATION")
    print("="*60)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print("\n📊 PERFORMANCE METRICS")
    print("-"*60)
    print(f"Training R² Score:   {train_r2:.4f} ({train_r2*100:.2f}%)")
    print(f"Testing R² Score:    {test_r2:.4f} ({test_r2*100:.2f}%)")
    print(f"Test RMSE:           ₹{test_rmse:,.2f}")
    print(f"Test MAE:            ₹{test_mae:,.2f}")

    # Check for overfitting
    print("\n🔍 OVERFITTING CHECK")
    print("-"*60)
    difference = train_r2 - test_r2
    print(f"Difference (Train R² - Test R²): {difference:.4f}")
    
    if difference > 0.1:
        print("⚠️  WARNING: Possible overfitting detected!")
        print("   Model performs much better on training than testing data.")
    elif difference > 0.05:
        print("⚡ Slight overfitting, but acceptable.")
    else:
        print("✅ Model generalizes well! No significant overfitting.")

    # Feature importance (coefficients)
    print("\n" + "="*60)
    print("STEP 5: FEATURE IMPORTANCE")
    print("="*60)
    print("\n📈 MODEL COEFFICIENTS (How much each feature affects rent):")
    print("-"*60)
    
    for feature, coef in zip(feature_columns, model.coef_):
        # Format positive/negative
        sign = "+" if coef >= 0 else ""
        print(f"{feature:20s}: {sign}₹{coef:>10,.2f}")
    
    print(f"{'Intercept':20s}:  ₹{model.intercept_:>10,.2f}")
    print("-"*60)
    
    print("\n💡 INTERPRETATION:")
    print(f"   • Every 1 sq ft adds: ₹{model.coef_[0]:.2f}")
    print(f"   • Every 1 BHK adds: ₹{model.coef_[1]:,.2f}")
    print(f"   • Every floor adds: ₹{model.coef_[2]:.2f}")
    print(f"   • Furnished adds: ₹{model.coef_[4]:,.2f} per level")
    print(f"   • Parking adds: ₹{model.coef_[5]:,.2f} per spot")
    print(f"   • Age reduces by: ₹{abs(model.coef_[6]):.2f} per year")
    
    return model, X_test, y_test, test_pred

def save_model(model):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model object
    """
    print("\n" + "="*60)
    print("STEP 6: SAVING MODEL")
    print("="*60)
    
    os.makedirs('models', exist_ok=True)
    model_path = 'models/rent_model.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Check file size
    file_size = os.path.getsize(model_path)
    print(f"✅ Model saved to: {model_path}")
    print(f"   File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")


def show_sample_predictions(X_test, y_test, predictions, df):
    """
    Display sample predictions vs actual values.
    
    Args:
        X_test: Test features
        y_test: Test targets
        predictions: Model predictions
        df: Original dataframe (for locality names)
    """
    print("\n" + "="*60)
    print("STEP 7: SAMPLE PREDICTIONS")
    print("="*60)
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Actual_Rent': y_test.values[:10],
        'Predicted_Rent': predictions[:10],
        'Error': y_test.values[:10] - predictions[:10],
        'Error_%': ((y_test.values[:10] - predictions[:10]) / y_test.values[:10] * 100)
    })
    
    print("\nFirst 10 test predictions:")
    print("-"*80)
    for i in range(min(10, len(comparison))):
        actual = comparison.iloc[i]['Actual_Rent']
        predicted = comparison.iloc[i]['Predicted_Rent']
        error = comparison.iloc[i]['Error']
        error_pct = comparison.iloc[i]['Error_%']
        
        # Color code based on accuracy
        if abs(error_pct) < 5:
            status = "✅ Excellent"
        elif abs(error_pct) < 10:
            status = "✓  Good"
        elif abs(error_pct) < 15:
            status = "~  Fair"
        else:
            status = "✗  Poor"
        
        print(f"Apt {i+1:2d}: Actual=₹{actual:>7,.0f} | Predicted=₹{predicted:>7,.0f} | "
              f"Error=₹{error:>6,.0f} ({error_pct:>5.1f}%) {status}")
    
    print("-"*80)
    
    # Overall accuracy
    avg_error_pct = abs(comparison['Error_%']).mean()
    print(f"\nAverage prediction error: {avg_error_pct:.2f}%")
    
    if avg_error_pct < 5:
        print("🌟 Outstanding accuracy!")
    elif avg_error_pct < 10:
        print("✅ Very good accuracy!")
    elif avg_error_pct < 15:
        print("👍 Good accuracy!")
    else:
        print("⚠️  Model needs improvement.")

def main():
    """
    Main function to run the entire training pipeline.
    """
    print("\n" + "🚀"*30)
    print("RENT PREDICTION MODEL TRAINING PIPELINE")
    print("🚀"*30)
    
    # Step 1: Load data
    df = load_data('data/apartments.csv')
    
    # Step 2: Preprocess
    df_processed, encoder = preprocess_data(df)
    
    # Step 3-5: Train and evaluate
    model, X_test, y_test, predictions = train_model(df_processed)
    
    # Step 6: Save model
    save_model(model)
    
    # Step 7: Show samples
    show_sample_predictions(X_test, y_test, predictions, df_processed)
    
    print("\n" + "="*60)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  ✓ models/rent_model.pkl (trained model)")
    print("  ✓ models/locality_encoder.pkl (locality encoder)")
    print("\nYou can now run the web app with: streamlit run app.py")
    print("="*60 + "\n")


# This runs when you execute the script
if __name__ == "__main__":
    main()