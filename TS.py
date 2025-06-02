import pickle
import pandas as pd
import numpy as np
import re
import csv
from io import StringIO
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, classification_report

def load_artifacts(artifacts_path=r'C:\Users\mosta\OneDrive\Desktop\Class final\preprocessing_artifacts.pkl', 
                  features_path=r'C:\Users\mosta\OneDrive\Desktop\Class final\chosen_features.pkl',
                  scaler_path=r'C:\Users\mosta\OneDrive\Desktop\Class final\preprocessing_artifacts.pkl'):
    with open(artifacts_path, 'rb') as f:
        preprocessing_artifacts = pickle.load(f)
    
    with open(features_path, 'rb') as f:
        chosen_features = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return preprocessing_artifacts, chosen_features, scaler

def load_model(model_path):
    """Load the trained model."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess_test_data(file_path, preprocessing_artifacts, chosen_features):
    """Preprocess a new CSV file using the saved artifacts."""
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Loaded data with shape: {df.shape}")
    
    if 'guest_satisfaction' in df.columns:
        target_map = {
            "Average": 0,
            "High": 1,
            "Very High": 2
        }
        df["guest_satisfaction"] = df["guest_satisfaction"].map(target_map)
    
    binary_cols = preprocessing_artifacts['binary_cols']
    le_dict = preprocessing_artifacts['le_dict']
    imputer = preprocessing_artifacts['imputer']
    date_cols = preprocessing_artifacts['date_cols']
    bad_binary_cols = preprocessing_artifacts['bad_binary_cols']
    tfidf_dict = preprocessing_artifacts['tfidf_dict']
    tfidf_scaler = preprocessing_artifacts['tfidf_scaler']
    pca = preprocessing_artifacts['pca']
    pca_columns = preprocessing_artifacts['pca_columns']
    amenity_weights = preprocessing_artifacts['amenity_weights']
    
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'t': 1, 'f': 0})
    print("Binary columns preprocessed")
    
    cols_to_convert = ['nightly_price', 'price_per_stay', 'security_deposit', 
                       'cleaning_fee', 'extra_people']
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace('[\\$,]', '', regex=True), errors='coerce')
            df[col] = df[col].fillna(0)
    
    if 'host_response_rate' in df.columns:
        df['host_response_rate'] = df['host_response_rate'].str.replace('%', '', regex=False)
        df['host_response_rate'] = pd.to_numeric(df['host_response_rate'], errors='coerce')
    print("Numeric columns preprocessed")
    
    cols_to_labelencode = ['property_type', 'room_type', 'cancellation_policy', 
                          'host_response_time', 'bed_type', 'neighbourhood_cleansed', 
                          'host_name', 'host_location', 'host_neighbourhood', 'street', 'zipcode']
    for col in cols_to_labelencode:
        if col in df.columns and col in le_dict:
            df[col] = df[col].astype(str)
            new_categories = set(df[col].unique()) - set(le_dict[col].classes_)
            if new_categories:
                print(f"Warning: Found {len(new_categories)} new categories in column '{col}'. Treating as unseen category.")
                df[col] = df[col].apply(lambda x: 'unseen_category' if x in new_categories else x)
                df[col] = df[col].apply(lambda x: le_dict[col].transform([x])[0] if x in le_dict[col].classes_ else -1)
            else:
                df[col] = le_dict[col].transform(df[col])
    print("Categorical columns preprocessed")
    
    columns_to_impute = ['host_response_rate', 'host_response_time']
    cols_present = [col for col in columns_to_impute if col in df.columns]
    if cols_present:
        df[cols_present] = imputer.transform(df[cols_present])
    print("Missing values imputed")
    
    date_pattern = r'^(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})$'
    
    def is_matching_format(series, pattern):
        """Check if series matches the given pattern."""
        sample = series.dropna().astype(str)
        if sample.empty:
            return False
        regex = re.compile(pattern)
        matches = sample.apply(lambda x: bool(regex.match(x)))
        return matches.mean() > 0.8
    
    found_date_cols = [col for col in df.columns if is_matching_format(df[col], date_pattern)]
    
    for col in found_date_cols:
        df[col] = pd.to_datetime(df[col], format='%m/%d/%Y', errors='coerce')
        df[f'{col}_year'] = df[col].dt.year.fillna(0)
        df[f'{col}_month'] = df[col].dt.month.fillna(0)
        df[f'{col}_dayofweek'] = df[col].dt.dayofweek.fillna(0)
    df.drop(columns=found_date_cols, inplace=True, errors='ignore')
    print("Date columns preprocessed")
    
    cols_to_drop = ['id', 'host_id', 'listing_url', 'host_url', 'square_feet', 
                   'neighbourhood', 'smart_location', 'state', 'market', 
                   'country_code', 'country', 'city', 'host_listings_count']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True, errors='ignore')
    print("Uninformative columns dropped")
    
    if 'zipcode' in df.columns:
        df['zipcode'] = df['zipcode'].astype(str)
    print("Rare values grouped")
    
    def clean_text(text):
        """Clean text data by removing special characters and normalizing."""
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s&-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    text_columns = ['notes', 'transit', 'access', 'interaction', 'house_rules',
                   'space', 'neighborhood_overview', 'description', 'summary', 
                   'name', 'host_about']
    
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].apply(clean_text).fillna("No information provided")
    
    tfidf_matrices = []
    for col in text_columns:
        if col in df.columns and col in tfidf_dict:
            tfidf_matrix = tfidf_dict[col].transform(df[col])
            tfidf_matrices.append(tfidf_matrix)
    
    if tfidf_matrices:
        X = hstack(tfidf_matrices)
        X_df = pd.DataFrame(X.toarray())
        
        X_scaled = tfidf_scaler.transform(X_df)
        X_pca = pca.transform(X_scaled)
        
        pca_df = pd.DataFrame(X_pca, columns=pca_columns)
        
        if 'guest_satisfaction' in df.columns:
            guest_satisfaction = df['guest_satisfaction'].reset_index(drop=True)
        else:
            guest_satisfaction = None
        
        df = df.drop(columns=[col for col in text_columns if col in df.columns], errors='ignore')
        
        df = pd.concat([df.reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)
        
        if guest_satisfaction is not None:
            df['guest_satisfaction'] = guest_satisfaction
    print("Text columns preprocessed")
    
    def tokenize_amenities(amenities_str):
        """Tokenize amenities string into a list."""
        if pd.isna(amenities_str) or not amenities_str.strip():
            return []
        clean_str = amenities_str.strip().strip('{}')
        if not clean_str:
            return []
        try:
            reader = csv.reader(StringIO(clean_str), quotechar='"', delimiter=',', skipinitialspace=True)
            tokens = next(reader, [])
            return [token.strip('"\'') for token in tokens if token.strip()]
        except Exception as e:
            print(f"Failed to parse: '{amenities_str}' - Error: {str(e)}")
            return []
    
    if 'amenities' in df.columns:
        df['amenities_tokenized'] = df['amenities'].apply(tokenize_amenities)
        
        def compute_amenity_score(amenities_list):
            return sum(amenity_weights.get(amenity, 0) for amenity in amenities_list)
        
        df['amenity_score'] = df['amenities_tokenized'].apply(compute_amenity_score)
    print("Amenities processed")
    
    df['host_total_listings_count'] = df['host_total_listings_count'].fillna(1)
    if 'host_neighbourhood' in df.columns and 'neighbourhood_cleansed' in df.columns:
        df['host_neighbourhood'] = df['host_neighbourhood'].fillna(df['neighbourhood_cleansed'])
    if 'maximum_nights' in df.columns:
        df['maximum_nights'] = np.clip(df['maximum_nights'], 1, 365)
    
    price_cols = ['nightly_price', 'minimum_nights', 'cleaning_fee', 'security_deposit', 
                  'extra_people', 'guests_included']
    if all(col in df.columns for col in price_cols):
        df['total_cost'] = (df['nightly_price'] * df['minimum_nights'] +
                           df['cleaning_fee'].fillna(0) +
                           df['security_deposit'].fillna(0) +
                           df['extra_people'].fillna(0) * df['guests_included'].fillna(0) * df['minimum_nights'])
    
    if 'number_of_stays' in df.columns:
        df['is_frequently_booked'] = df['number_of_stays'] > df['number_of_stays'].mean()
    
    room_cols = ['bedrooms', 'bathrooms']
    if all(col in df.columns for col in room_cols):
        df[room_cols] = df[room_cols].fillna(1)
    
    if 'beds' in df.columns and 'bedrooms' in df.columns:
        df['beds'] = df['beds'].fillna(df['bedrooms'])
    
    if 'bedrooms' in df.columns and 'bathrooms' in df.columns and 'accommodates' in df.columns:
        df['space_to_people_ratio'] = (df['bedrooms'] + df['bathrooms']) / df['accommodates']
    
    if 'host_is_superhost' in df.columns:
        df['host_is_superhost'] = df['host_is_superhost'].fillna(1)
    print("Feature engineering completed")
    
    available_features = [f for f in chosen_features if f in df.columns]
    if 'guest_satisfaction' in df.columns:
        df_final = df[available_features + ['guest_satisfaction']]
    else:
        df_final = df[available_features]
    
    print(f"Final features selected: {len(available_features)} out of {len(chosen_features)} chosen features")
    
    return df_final

def main():
    test_csv_path = r"C:\Users\mosta\OneDrive\Desktop\Class final\GuestSatisfactionPrediction_test_Classification.csv"  # Replace with your actual test data path
    
    preprocessing_artifacts, chosen_features, scaler = load_artifacts()
    print("Loaded preprocessing artifacts")
    
    # model = load_model(r"C:\Users\mosta\OneDrive\Desktop\Class final\best_random_forest_model_Classification.pkl")
    # model = load_model(r"C:\Users\mosta\OneDrive\Desktop\Class final\best_xgboost_model_Classification.pkl")
    # model = load_model(r"C:\Users\mosta\OneDrive\Desktop\Class final\best_logistic_regression_model_Classification.pkl")
    model = load_model(r"C:\Users\mosta\OneDrive\Desktop\Class final\best_catboost_model_Classification.pkl")
    print("Loaded model")
    
    df_processed = preprocess_test_data(test_csv_path, preprocessing_artifacts, chosen_features)
    print(f"Processed data shape: {df_processed.shape}")
    df_processed.info()
    prediction_map = {0: "Average", 1: "High", 2: "Very High"}

    if 'guest_satisfaction' not in df_processed.columns:
        X_test = df_processed
        X_test_scaled = scaler.transform(X_test)

        predictions = model.predict(X_test_scaled)
        text_predictions = [prediction_map[pred] for pred in predictions]

        result_df = pd.DataFrame({
            'predicted_satisfaction': predictions,
            'predicted_satisfaction_text': text_predictions
        })

        result_df.to_csv("predictions.csv", index=False)
        print("Predictions saved to 'predictions.csv'")

    else:
        X_test = df_processed.drop('guest_satisfaction', axis=1)
        y_test = df_processed['guest_satisfaction']
        # X_test_scaled = scaler.transform(X_test)

        predictions = model.predict(X_test)
        text_predictions = [prediction_map[pred] for pred in predictions]
        actual_text = [prediction_map[actual] for actual in y_test]
 
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)

        print(f"Testing Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)

        result_df = pd.DataFrame({
            'actual_satisfaction': y_test,
            'actual_satisfaction_text': actual_text,
            'predicted_satisfaction': predictions,
            'predicted_satisfaction_text': text_predictions
        })

        result_df.to_csv("predictions_with_actual.csv", index=False)
        print("Predictions with actual values saved to 'predictions_with_actual.csv'")


if __name__ == "__main__":
    main()
