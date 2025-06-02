import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler

def load_artifacts():
    
    with open('preprocessing_artifacts.pkl', 'rb') as f:
        preprocessing_artifacts = pickle.load(f)
    
    models = {}
    for model_name in ['random_forest', 'polynomial', 'xgboost', 'catboost']:
        with open(f'best_{model_name}_model.pkl', 'rb') as f:
            models[model_name] = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('chosen_features.pkl', 'rb') as f:
        chosen_features = pickle.load(f)
    
    return preprocessing_artifacts, models, scaler, chosen_features

def preprocess_binary_columns(df, binary_cols):
  
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'t': 1, 'f': 0}).fillna(0)
    return df

def preprocess_numeric_columns(df):
    
    cols_to_convert = ['nightly_price', 'price_per_stay', 'security_deposit', 
                       'cleaning_fee', 'extra_people']
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace('[\\$,]', '', regex=True), errors='coerce').fillna(0)
    
    if 'host_response_rate' in df.columns:
        df['host_response_rate'] = df['host_response_rate'].str.replace('%', '', regex=False)
        df['host_response_rate'] = pd.to_numeric(df['host_response_rate'], errors='coerce').fillna(0)
    return df

def preprocess_categorical_columns(df, ohe_dict, cat_scaler, cat_pca, cat_pca_columns):
   
    cols_to_encode = ['property_type', 'room_type', 'cancellation_policy', 
                      'host_response_time', 'bed_type', 'neighbourhood_cleansed', 
                      'host_name', 'host_location']
    
    for col in cols_to_encode:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
        else:
            df[col] = 'Unknown'
    
    ohe = ohe_dict['encoder']
    X_cat = ohe.transform(df[cols_to_encode])
    X_cat_dense = X_cat.toarray()
    X_cat_scaled = cat_scaler.transform(X_cat_dense)
    X_pca = cat_pca.transform(X_cat_scaled)
    X_pca_df = pd.DataFrame(X_pca, columns=cat_pca_columns)
    
    df_preprocessed = pd.concat([df.reset_index(drop=True), X_pca_df.reset_index(drop=True)], axis=1)
    df_preprocessed = df_preprocessed.drop(columns=cols_to_encode, errors='ignore')
    return df_preprocessed

def impute_missing_values(df, imputer, columns_to_impute=['host_response_rate']):
   
    for col in columns_to_impute:
        if col not in df.columns:
            df[col] = 0
    df[columns_to_impute] = imputer.transform(df[columns_to_impute])
    return df

def preprocess_dates(df, date_cols):
    
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%m/%d/%Y', errors='coerce')
            df[f'{col}_year'] = df[col].dt.year.fillna(0)
            df[f'{col}_month'] = df[col].dt.month.fillna(0)
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek.fillna(0)
        else:
            df[f'{col}_year'] = 0
            df[f'{col}_month'] = 0
            df[f'{col}_dayofweek'] = 0
    df.drop(columns=[col for col in date_cols if col in df.columns], inplace=True)
    return df

def drop_uninformative_columns(df, bad_binary_cols):
    
    df = df.drop(columns=bad_binary_cols, errors='ignore')
    cols_to_drop = ['id', 'host_id', 'listing_url', 'host_url', 'square_feet', 
                    'neighbourhood', 'smart_location', 'state', 'market', 
                    'country_code', 'country', 'city', 'host_neighbourhood', 
                    'host_listings_count']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    return df

def group_rare_values(df, column_name='zipcode'):
    
    if column_name in df.columns:
        df[column_name] = df[column_name].astype(str).fillna('Other')
    else:
        df[column_name] = 'Other'
    return df

def clean_text(text):
   
    if pd.isna(text):
        return ""
    import re
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s&-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text_columns(df, tfidf_dict, tfidf_scaler, pca, pca_columns):
    
    text_columns = ['notes', 'transit', 'access', 'interaction', 'house_rules',
                    'space', 'neighborhood_overview', 'description', 'summary', 
                    'name', 'host_about']
    
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].apply(clean_text).fillna("No information provided")
        else:
            df[column] = "No information provided"
    
    tfidf_matrices = []
    for col in text_columns:
        tfidf = tfidf_dict[col]
        tfidf_matrix = tfidf.transform(df[col])
        tfidf_matrices.append(tfidf_matrix)
    
    X = hstack(tfidf_matrices)
    X_df = pd.DataFrame(X.toarray(), columns=[f"{col}_{word}" for col in text_columns for word in tfidf_dict[col].get_feature_names_out()])
    X_scaled = tfidf_scaler.transform(X_df)
    X_pca = pca.transform(X_scaled)
    X_pca_df = pd.DataFrame(X_pca, columns=pca_columns)
    
    df_preprocessed = pd.concat([df.reset_index(drop=True), X_pca_df.reset_index(drop=True)], axis=1)
    df_preprocessed = df_preprocessed.drop(columns=text_columns, errors='ignore')
    return df_preprocessed

def tokenize_amenities(amenities_str):
    
    if pd.isna(amenities_str) or not amenities_str.strip():
        return []
    from io import StringIO
    import csv
    clean_str = amenities_str.strip().strip('{}')
    if not clean_str:
        return []
    try:
        reader = csv.reader(StringIO(clean_str), quotechar='"', delimiter=',', skipinitialspace=True)
        tokens = next(reader, [])
        return [token.strip('"\'') for token in tokens if token.strip()]
    except Exception:
        return []

def compute_amenity_scores(df, amenity_weights):
   
    df['amenities_tokenized'] = df['amenities'].apply(tokenize_amenities)
    df['amenity_score'] = df['amenities_tokenized'].apply(lambda x: sum(amenity_weights.get(amenity, 0) for amenity in x))
    return df

def feature_engineering(df):
    """Apply feature engineering as done in training."""
    df['host_total_listings_count'] = df['host_total_listings_count'].fillna(1)
    df['host_is_superhost'] = df['host_is_superhost'].fillna(1)
    df['host_identity_verified'] = df['host_is_superhost'].fillna(1)
    df['maximum_nights'] = np.clip(df['maximum_nights'].fillna(365), 1, 365)
    
    for col in ['nightly_price', 'cleaning_fee', 'security_deposit', 'extra_people', 'guests_included', 'minimum_nights']:
        if col not in df.columns:
            df[col] = 0
    
    df['total_cost'] = (df['nightly_price'] * df['minimum_nights'] +
                        df['cleaning_fee'] +
                        df['security_deposit'] +
                        df['extra_people'] * df['guests_included'] * df['minimum_nights'])
    
    if 'number_of_stays' in df.columns:
        df['is_frequently_booked'] = df['number_of_stays'] > df['number_of_stays'].mean()
    else:
        df['is_frequently_booked'] = False
    
    df[['bedrooms', 'bathrooms']] = df[['bedrooms', 'bathrooms']].fillna(1)
    df['beds'] = df['beds'].fillna(df['bedrooms'])
    df['space_to_people_ratio'] = (df['bedrooms'] + df['bathrooms']) / df['accommodates'].fillna(1)
    return df

def preprocess_test_data(df, preprocessing_artifacts, chosen_features):
   
    df = preprocess_binary_columns(df, preprocessing_artifacts['binary_cols'])
    df = preprocess_numeric_columns(df)
    df = preprocess_categorical_columns(df, preprocessing_artifacts['ohe_dict'], 
                                       preprocessing_artifacts['cat_scaler'], 
                                       preprocessing_artifacts['cat_pca'], 
                                       preprocessing_artifacts['cat_pca_columns'])
    df = impute_missing_values(df, preprocessing_artifacts['imputer'])
    df = preprocess_dates(df, preprocessing_artifacts['date_cols'])
    df = drop_uninformative_columns(df, preprocessing_artifacts['bad_binary_cols'])
    df = group_rare_values(df, 'zipcode')
    df = preprocess_text_columns(df, preprocessing_artifacts['tfidf_dict'], 
                                preprocessing_artifacts['tfidf_scaler'], 
                                preprocessing_artifacts['pca'], 
                                preprocessing_artifacts['pca_columns'])
    df = compute_amenity_scores(df, preprocessing_artifacts['amenity_weights'])
    df = feature_engineering(df)
    
  
    for col in chosen_features:
        if col not in df.columns:
            df[col] = 0
    
    return df[chosen_features]

def main():
   
    preprocessing_artifacts, models, scaler, chosen_features = load_artifacts()
    
  
    test_df = pd.read_csv('GuestSatisfactionPrediction_test_Reg.csv', low_memory=False)
    
   
    X_test = preprocess_test_data(test_df, preprocessing_artifacts, chosen_features)
    
  
    cat_features = [f for f in chosen_features[:4] if f in X_test.columns]
    for col in cat_features:
        X_test[col] = X_test[col].astype(int)
    
    X_test_scaled = scaler.transform(X_test)
    
 
    results = []
    if 'review_scores_rating' in test_df.columns:
        y_test = test_df['review_scores_rating'].fillna(test_df['review_scores_rating'].median())
    else:
        y_test = None
    
    for model_name, model in models.items():
      
        if model_name in ['polynomial']:
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
        
       
        predictions_df = pd.DataFrame({
            'Prediction': y_pred
        })
        predictions_df.to_csv(f'{model_name}_predictions.csv', index=False)
        
      
        if y_test is not None:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"{model_name} - Mean Squared Error: {mse:.2f}")
            print(f"{model_name} - R-squared: {r2:.2f}")
            results.append({
                'Model': model_name,
                'MSE': mse,
                'R2': r2
            })
    
   
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('regression_results.csv', index=False)
        print("\nSummary of Results:")
        print(results_df)

if __name__ == "__main__":
    main()