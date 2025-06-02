import re
import csv
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import StringIO
from collections import defaultdict
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb
from catboost import CatBoostRegressor
import torch

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_info_columns', 1000)
pd.set_option('display.max_info_rows', 1000)

def load_data(file_path='GuestSatisfactionPrediction.csv'):
    return pd.read_csv(file_path, low_memory=False)

# Binary columns l 0 , 1 (one hot encoding el 8laba) 
def preprocess_binary_columns(df):
    binary_cols = [col for col in df.columns 
                   if df[col].dropna().nunique() <= 2 and 
                   set(df[col].dropna().unique())]
    for col in binary_cols:
        df[col] = df[col].map({'t': 1, 'f': 0})
    return df, binary_cols

#tyr ay dollar sign w precent w ay haga fe numeric columns
def preprocess_numeric_columns(df):
    cols_to_convert = ['nightly_price', 'price_per_stay', 'security_deposit', 
                       'cleaning_fee', 'extra_people']
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col].str.replace('[\\$,]', '', regex=True), errors='coerce')
        df[col] = df[col].fillna(0)
    
    df['host_response_rate'] = df['host_response_rate'].str.replace('%', '', regex=False)
    df['host_response_rate'] = pd.to_numeric(df['host_response_rate'], errors='coerce')
    return df


# categor bl ohe last edit akhr mra bgd 
def preprocess_categorical_columns(df):
    """Encode categorical columns using OneHotEncoder and apply PCA to limit to 150 columns."""
    ohe_dict = {}
    cols_to_encode = ['property_type', 'room_type', 'cancellation_policy', 
                     'host_response_time', 'bed_type', 'neighbourhood_cleansed', 
                     'host_name', 'host_location']
    
    
    for col in cols_to_encode:
        df[col] = df[col].fillna('Unknown').astype(str)
    
  
    ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    X_cat = ohe.fit_transform(df[cols_to_encode])
    ohe_dict['encoder'] = ohe
    
    
    feature_names = ohe.get_feature_names_out(cols_to_encode)
    
   
    X_cat_dense = X_cat.toarray()
    
    
    scaler = StandardScaler()
    X_cat_scaled = scaler.fit_transform(X_cat_dense)
    
    
    n_components = min(150, X_cat_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_cat_scaled)
    
    
    pca_columns = [f'cat_pca_{i+1}' for i in range(n_components)]
    
   
    X_pca_df = pd.DataFrame(X_pca, columns=pca_columns)
    
    
    df_preprocessed = pd.concat([df.reset_index(drop=True), X_pca_df.reset_index(drop=True)], axis=1)
    df_preprocessed = df_preprocessed.drop(columns=cols_to_encode, errors='ignore')
    
    return df_preprocessed, ohe_dict, scaler, pca, pca_columns

#el hindi guy ele 4ar7 el knnimputer
def impute_missing_values(df, columns_to_impute=['host_response_rate']):
    imputer = KNNImputer(n_neighbors=5)
    df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
    return df, imputer

#2sm el date m y d
def preprocess_dates(df):
    date_pattern = r'^(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})$'
    date_cols = [col for col in df.columns if is_matching_format(df[col], date_pattern)]
    
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format='%m/%d/%Y', errors='coerce')
        df[f'{col}_year'] = df[col].dt.year.fillna(0)
        df[f'{col}_month'] = df[col].dt.month.fillna(0)
        df[f'{col}_dayofweek'] = df[col].dt.dayofweek.fillna(0)
    df.drop(columns=date_cols, inplace=True)
    return df, date_cols

# el match bta3t regex bta3t el date 
def is_matching_format(series, pattern):
    sample = series.dropna().astype(str)
    regex = re.compile(pattern)
    matches = sample.apply(lambda x: bool(regex.match(x)))
    return matches.mean() > 0.8

def drop_uninformative_columns(df):
    df = df.dropna(axis=1, how='all')
    
    binary_columns = [col for col in df.columns if df[col].nunique() <= 2]
    dominance_threshold = 95
    bad_binary_columns = [col for col in binary_columns 
                         if (df[col].value_counts(normalize=True) * 100 > dominance_threshold).any()]
    df.drop(columns=bad_binary_columns, inplace=True)
    
    cols_to_drop = ['id', 'host_id', 'listing_url', 'host_url', 'square_feet', 
                    'neighbourhood', 'smart_location', 'state', 'market', 
                    'country_code', 'country', 'city','host_neighbourhood','host_listings_count']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    return df, bad_binary_columns

#ay minumum unique values a2l mn 10 3shan el encoder myt3bsh w yzwd columns brdu
def group_rare_values(df, column_name, min_count=10):
    if column_name in df.columns:
        if column_name == 'zipcode':
            df[column_name] = df[column_name].astype(str)
        value_counts = df[column_name].value_counts()
        rare_values = value_counts[value_counts < min_count].index
        df[column_name] = df[column_name].apply(lambda x: 'Other' if x in rare_values else x)
    return df


# shwyt cleaning mknsa
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s&-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text



#khls el klam el tf rakm 1 
def preprocess_text_columns(df):
    text_columns = ['notes', 'transit', 'access', 'interaction', 'house_rules',
                    'space', 'neighborhood_overview', 'description', 'summary', 
                    'name', 'host_about']
    
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].apply(clean_text).fillna("No information provided")
    
    max_features = {col: 100 for col in text_columns}
    max_features['notes'] = 50
    
    tfidf_dict = {}
    tfidf_matrices = []
    feature_names = []
    for col in text_columns:
        if col in df.columns:
            tfidf = TfidfVectorizer(max_features=max_features[col], stop_words='english')
            tfidf_matrix = tfidf.fit_transform(df[col])
            tfidf_matrices.append(tfidf_matrix)
            tfidf_dict[col] = tfidf
            feature_names.extend([f"{col}_{word}" for word in tfidf.get_feature_names_out()])
    
    X = hstack(tfidf_matrices)
    X_df = pd.DataFrame(X.toarray(), columns=feature_names)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    
    n_components = 200
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_columns = [f'tfidf_pca_{i+1}' for i in range(n_components)]
    X_pca_df = pd.DataFrame(X_pca, columns=pca_columns)
    
    df_preprocessed = pd.concat([df.reset_index(drop=True), X_pca_df.reset_index(drop=True)], axis=1)
    df_preprocessed = df_preprocessed.drop(columns=[col for col in text_columns if col in df_preprocessed.columns], errors='ignore')
    return df_preprocessed, tfidf_dict, scaler, pca, pca_columns

#tokenize 3shan yb2a same pattern
def tokenize_amenities(amenities_str):
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

#weighted encoding 
def compute_amenity_scores(df):
    df['amenities_tokenized'] = df['amenities'].apply(tokenize_amenities)
    all_amenities = df['amenities_tokenized'].explode().dropna()
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
    unique_amenities = list(set(all_amenities))
    results = []
    for amenity in unique_amenities:
        has_amenity = df_train['amenities_tokenized'].apply(lambda x: amenity in x)
        if has_amenity.sum() == 0:
            continue
        mean_with = df_train[has_amenity]['review_scores_rating'].mean()
        mean_without = df_train[~has_amenity]['review_scores_rating'].mean()
        results.append({
            'amenity': amenity,
            'mean_with': mean_with,
            'mean_without': mean_without,
            'mean_diff': mean_with - mean_without,
            'count': has_amenity.sum()
        })
    
    amenity_effects = pd.DataFrame(results)
    scaler = MinMaxScaler()
    amenity_effects['normalized_weight'] = scaler.fit_transform(amenity_effects[['mean_diff']])
    amenity_weights = dict(zip(amenity_effects['amenity'], amenity_effects['normalized_weight']))
    
    def compute_amenity_score(amenities_list):
        return sum(amenity_weights.get(amenity, 0) for amenity in amenities_list)
    
    df_train['amenity_score'] = df_train['amenities_tokenized'].apply(compute_amenity_score)
    df_test['amenity_score'] = df_test['amenities_tokenized'].apply(compute_amenity_score)
    df = pd.concat([df_train, df_test], axis=0).sort_index()
    return df, amenity_weights, scaler



def feature_engineering(df):

   
    df['host_total_listings_count'] = df['host_total_listings_count'].fillna(1)
    df['host_is_superhost'] = df['host_is_superhost'].fillna(1)
    df['host_identity_verified'] = df['host_is_superhost'].fillna(1)

    # df['host_neighbourhood'] = df['host_neighbourhood'].fillna(df['neighbourhood_cleansed'])
    df['maximum_nights'] = np.clip(df['maximum_nights'], 1, 365)
    df['total_cost'] = (df['nightly_price'] * df['minimum_nights'] +
                        df['cleaning_fee'].fillna(0) +
                        df['security_deposit'].fillna(0) +
                        df['extra_people'].fillna(0) * df['guests_included'].fillna(0) * df['minimum_nights'])
    df['is_frequently_booked'] = df['number_of_stays'] > df['number_of_stays'].mean()
    df[['bedrooms', 'bathrooms']] = df[['bedrooms', 'bathrooms']].fillna(1)
    df['beds'] = df['beds'].fillna(df['bedrooms'])
    df['space_to_people_ratio'] = (df['bedrooms'] + df['bathrooms']) / df['accommodates']
   

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_info_columns', 1000)
    pd.set_option('display.max_info_rows', 1000)
    df.info(verbose=True, show_counts=True)
    return df

#anova / corr
def select_features(df, cat_pca_columns=None):
   
   
    discrete_columns = [col for col in df.columns if df[col].nunique() < 20]

    excluded_categorical = ['property_type', 'neighbourhood_cleansed', 'zipcode', 
                           'host_neighbourhood', 'host_location', 'street', 
                           'room_type', 'cancellation_policy', 'host_response_time', 
                           'bed_type', 'host_name']
    discrete_columns = [col for col in discrete_columns if col not in excluded_categorical]
    
    anova_results = []
    for col in discrete_columns:
        try:
            groups = [group['review_scores_rating'].dropna().values 
                      for _, group in df.groupby(col) if len(group) > 1]
            if len(groups) >= 2:
                f_val, p_val = f_oneway(*groups)
                anova_results.append({'feature': col, 'f_statistic': f_val, 'p_value': p_val})
        except Exception as e:
            print(f"Skipping {col} due to error: {str(e)}")
    
    anova_df = pd.DataFrame(anova_results).sort_values(by='f_statistic', ascending=False)
    top_discrete_features = anova_df.nlargest(4, 'f_statistic')['feature'].tolist()
    
    
    numeric_df = df.select_dtypes(include=['number'])
    non_discrete_df = numeric_df.loc[:, ~numeric_df.columns.isin(discrete_columns)]
    correlation_matrix = non_discrete_df.corr()
    correlation_with_target = correlation_matrix['review_scores_rating'].abs().drop('review_scores_rating', errors='ignore')
    top_continuous_features = correlation_with_target.nlargest(4).index.tolist()
    
    return top_discrete_features + top_continuous_features

#el models na shlt el svr 3shan by7ml fe 140 min !
def train_and_evaluate_model(model, param_grid, X_train, X_test, y_train, y_test, model_name, scaler=None):
    feature_names = X_train.columns.tolist()
    
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=2)
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    grid_search.fit(X_train, y_train)
    
    print(f"Best Hyperparameters for {model_name}: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    
    with open(f'best_{model_name}_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name} - Mean Squared Error: {mse:.2f}")
    print(f"{model_name} - R-squared: {r2:.2f}")
    return best_model, feature_names



def visualize_important_features(df, chosen_features, amenity_weights=None, top_n_amenities=10):
   
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # Distribution of Review Scores
    plt.figure(figsize=(10, 6))
    sns.histplot(df['review_scores_rating'].dropna(), bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Review Scores', fontsize=14)
    plt.xlabel('Review Score (1-100)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Feature Importance (using RandomForest)
    X = df[chosen_features].fillna(0)  
    y = df['review_scores_rating'].fillna(df['review_scores_rating'].median())
    rf = RandomForestRegressor(random_state=42, n_estimators=100)
    rf.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'Feature': chosen_features,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, color='lightcoral')
    plt.title('Feature Importance (Random Forest)', fontsize=14)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    #  Correlation Heatmap
    continuous_features = [f for f in chosen_features if f in df.select_dtypes(include=['number']).columns]
    if continuous_features:
        corr_df = df[continuous_features + ['review_scores_rating']].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation Heatmap of Top Continuous Features', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    # Top Amenities Impact
    if amenity_weights:
        amenity_df = pd.DataFrame({
            'Amenity': list(amenity_weights.keys()),
            'Weight': list(amenity_weights.values())
        }).sort_values(by='Weight', ascending=False).head(top_n_amenities)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Weight', y='Amenity', data=amenity_df, color='seagreen')
        plt.title(f'Top {top_n_amenities} Amenities by Impact on Review Scores', fontsize=14)
        plt.xlabel('Normalized Weight', fontsize=12)
        plt.ylabel('Amenity', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    # Box Plots for Top Continuous Features
    if continuous_features:
        top_continuous = continuous_features[:4] 
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(top_continuous, 1):
            plt.subplot(2, 2, i)
            sns.boxplot(y=df[col].dropna(), color='lightblue')
            plt.title(f'Distribution of {col}', fontsize=12)
            plt.ylabel(col, fontsize=10)
        plt.tight_layout()
        plt.show()



def main():
    df = load_data()
    df, binary_cols = preprocess_binary_columns(df)
    df = preprocess_numeric_columns(df)
    df, ohe_dict, cat_scaler, cat_pca, cat_pca_columns = preprocess_categorical_columns(df)
    df, imputer = impute_missing_values(df)
    df, date_cols = preprocess_dates(df)
    df, bad_binary_cols = drop_uninformative_columns(df)
    df = group_rare_values(df, 'zipcode')
    df, tfidf_dict, tfidf_scaler, pca, pca_columns = preprocess_text_columns(df)
    df, amenity_weights, amenity_scaler = compute_amenity_scores(df)
    df = feature_engineering(df)
    
   
    df.to_csv('airbnb_preprocessed_tfidf_pca.csv', index=False)
    df = pd.read_csv('airbnb_preprocessed_tfidf_pca.csv', low_memory=False)
    
   
    preprocessing_artifacts = {
        'binary_cols': binary_cols,
        'ohe_dict': ohe_dict,
        'cat_scaler': cat_scaler,
        'cat_pca': cat_pca,
        'cat_pca_columns': cat_pca_columns,
        'imputer': imputer,
        'date_cols': date_cols,
        'bad_binary_cols': bad_binary_cols,
        'tfidf_dict': tfidf_dict,
        'tfidf_scaler': tfidf_scaler,
        'pca': pca,
        'pca_columns': pca_columns,
        'amenity_weights': amenity_weights,
        'amenity_scaler': amenity_scaler
    }
    with open('preprocessing_artifacts.pkl', 'wb') as f:
        pickle.dump(preprocessing_artifacts, f)
    

  
    chosen_features = select_features(df)
    visualize_important_features(df, chosen_features, amenity_weights=amenity_weights)
    with open('chosen_features.pkl', 'wb') as f:
        pickle.dump(chosen_features, f)
    
    model_df = df[[c for c in chosen_features if c in df.columns] + ['review_scores_rating']].copy()
    model_df[chosen_features[:4]] = model_df[chosen_features[:4]].astype(int)
    
    X = model_df.drop(columns=['review_scores_rating'])
    y = model_df['review_scores_rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    models = [
        (
            RandomForestRegressor(random_state=42),
            {
                'n_estimators': [100, 300, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5]
            },
            "random_forest"
        ),
        (
            Pipeline([('poly', PolynomialFeatures()), ('regressor', LinearRegression())]),
            {
                'poly__degree': [1, 2, 3, 4, 5],
                'regressor__fit_intercept': [True, False]
            },
            "polynomial"
        ),
        (
            xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
            {
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [4, 6, 8],
                'n_estimators': [100, 300, 500]
            },
            "xgboost"
        ),
        (
            CatBoostRegressor(cat_features=chosen_features[:4], silent=True, random_state=42),
            {
                'iterations': [500, 1000, 1500],
                'learning_rate': [0.01, 0.05, 0.1],
                'depth': [6, 8, 10]
            },
            "catboost"
        )
        # ,
        # (
        #     Pipeline([('poly', PolynomialFeatures()), ('regressor', SVR())]),
        #     {
        #         'poly__degree': [1, 2],
        #         'regressor__kernel': ['rbf', 'poly'],
        #         'regressor__C': [0.1, 1, 10],
        #         'regressor__epsilon': [0.1, 0.2],
        #         'regressor__degree': [2, 3],
        #         'regressor__gamma': ['scale']
        #     },
        #     "svr"
        # )
    ]
    
    for model, param_grid, model_name in models:
        best_model, feature_names = train_and_evaluate_model(model, param_grid, X_train, X_test, y_train, y_test, model_name, scaler if model_name in ['polynomial', 'svr'] else None)
        print(f"Saved model {model_name} with features: {feature_names}")

if __name__ == "__main__":
    main()