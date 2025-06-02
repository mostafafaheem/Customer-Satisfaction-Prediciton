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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from scipy.sparse import hstack
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
from scipy.stats import f_oneway, chi2_contingency

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_info_columns', 1000)
pd.set_option('display.max_info_rows', 1000)

def load_data(file_path=r"C:\Users\mosta\OneDrive\Desktop\Semester 6\Machine Learning\Project\MS2\Datasets\GuestSatisfactionPredictionMilestone2.csv"):
    """Load the dataset."""
    df= pd.read_csv(file_path, low_memory=False)
    target_map = {
    "Average": 0,
    "High": 1,
    "Very High": 2
    }
    df["guest_satisfaction"] = df["guest_satisfaction"].map(target_map)
    return df

def preprocess_binary_columns(df):
    """Convert binary columns ('t', 'f') to (1, 0)."""
    binary_cols = [col for col in df.columns 
                   if df[col].dropna().nunique() <= 2 and 
                   set(df[col].dropna().unique())]
    for col in binary_cols:
        df[col] = df[col].map({'t': 1, 'f': 0})
    return df, binary_cols

def preprocess_numeric_columns(df):
    """Convert price-related columns to numeric and handle missing values."""
    cols_to_convert = ['nightly_price', 'price_per_stay', 'security_deposit', 
                       'cleaning_fee', 'extra_people']
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col].str.replace('[\\$,]', '', regex=True), errors='coerce')
        df[col] = df[col].fillna(0)
    
    df['host_response_rate'] = df['host_response_rate'].str.replace('%', '', regex=False)
    df['host_response_rate'] = pd.to_numeric(df['host_response_rate'], errors='coerce')
    return df

def preprocess_categorical_columns(df):
    """Encode categorical columns using LabelEncoder."""
    le_dict = {}
    cols_to_labelencode = ['property_type', 'room_type', 'cancellation_policy', 
                           'host_response_time', 'bed_type', 'neighbourhood_cleansed', 
                           'host_name', 'host_location', 'host_neighbourhood', 'street', 'zipcode']
    for col in cols_to_labelencode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    return df, le_dict

def impute_missing_values(df, columns_to_impute=['host_response_rate', 'host_response_time']):
    """Impute missing values using KNNImputer."""
    imputer = KNNImputer(n_neighbors=5)
    df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
    return df, imputer

def preprocess_dates(df):
    """Convert date columns to datetime and extract features."""
    date_pattern = r'^(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})$'
    date_cols = [col for col in df.columns if is_matching_format(df[col], date_pattern)]
    
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format='%m/%d/%Y', errors='coerce')
        df[f'{col}_year'] = df[col].dt.year.fillna(0)
        df[f'{col}_month'] = df[col].dt.month.fillna(0)
        df[f'{col}_dayofweek'] = df[col].dt.dayofweek.fillna(0)
    df.drop(columns=date_cols, inplace=True)
    return df, date_cols

def is_matching_format(series, pattern):
    """Check if series matches the given pattern."""
    sample = series.dropna().astype(str)
    regex = re.compile(pattern)
    matches = sample.apply(lambda x: bool(regex.match(x)))
    return matches.mean() > 0.8

def drop_uninformative_columns(df):
    """Drop columns with all NaN or highly dominant binary values."""
    df = df.dropna(axis=1, how='all')
    
    binary_columns = [col for col in df.columns if df[col].nunique() <= 2]
    dominance_threshold = 95
    bad_binary_columns = [col for col in binary_columns 
                         if (df[col].value_counts(normalize=True) * 100 > dominance_threshold).any()]
    df.drop(columns=bad_binary_columns, inplace=True)
    
    cols_to_drop = ['id', 'host_id', 'listing_url', 'host_url', 'square_feet', 
                    'neighbourhood', 'smart_location', 'state', 'market', 
                    'country_code', 'country', 'city',"host_listings_count"]
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    return df, bad_binary_columns

def group_rare_values(df, column_name, min_count=10):
    """Group rare values in a column."""
    if column_name in df.columns:
        if column_name == 'zipcode':
            df[column_name] = df[column_name].astype(str)
        value_counts = df[column_name].value_counts()
        rare_values = value_counts[value_counts < min_count].index
        df[column_name] = df[column_name].apply(lambda x: 'Other' if x in rare_values else x)
    return df

def clean_text(text):
    """Clean text data by removing special characters and normalizing."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s&-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text_columns(df):
    """Apply TF-IDF and PCA to text columns."""
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
    
    n_components = 40
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_columns = [f'tfidf_pca_{i+1}' for i in range(n_components)]
    X_pca_df = pd.DataFrame(X_pca, columns=pca_columns)
    
    # Save target column before modifications
    if 'guest_satisfaction' in df.columns:
        guest_satisfaction = df['guest_satisfaction'].reset_index(drop=True)
    else:
        raise ValueError("'guest_satisfaction' column missing before text processing!")

    df = df.drop(columns=[col for col in text_columns if col in df.columns], errors='ignore')

    # Combine cleaned DF with PCA features
    df_preprocessed = pd.concat([df.reset_index(drop=True), X_pca_df.reset_index(drop=True)], axis=1)

    # Reattach guest_satisfaction
    df_preprocessed['guest_satisfaction'] = guest_satisfaction

    return df_preprocessed, tfidf_dict, scaler, pca, pca_columns

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

def compute_amenity_scores(df):
    """Compute amenity scores based on their impact on guest_satisfaction."""
    df['amenities_tokenized'] = df['amenities'].apply(tokenize_amenities)
    all_amenities = df['amenities_tokenized'].explode().dropna()
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
    unique_amenities = list(set(all_amenities))
    results = []
    for amenity in unique_amenities:
        has_amenity = df_train['amenities_tokenized'].apply(lambda x: amenity in x)
        if has_amenity.sum() == 0:
            continue
        mean_with = df_train[has_amenity]['guest_satisfaction'].mean()
        mean_without = df_train[~has_amenity]['guest_satisfaction'].mean()
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
    """Perform feature engineering."""
    # df['host_listings_count'] = df['host_listings_count'].fillna(1)
    df['host_total_listings_count'] = df['host_total_listings_count'].fillna(1)
    df['host_neighbourhood'] = df['host_neighbourhood'].fillna(df['neighbourhood_cleansed'])
    df['maximum_nights'] = np.clip(df['maximum_nights'], 1, 365)
    df['total_cost'] = (df['nightly_price'] * df['minimum_nights'] +
                        df['cleaning_fee'].fillna(0) +
                        df['security_deposit'].fillna(0) +
                        df['extra_people'].fillna(0) * df['guests_included'].fillna(0) * df['minimum_nights'])
    df['is_frequently_booked'] = df['number_of_stays'] > df['number_of_stays'].mean()
    df[['bedrooms', 'bathrooms']] = df[['bedrooms', 'bathrooms']].fillna(1)
    df['beds'] = df['beds'].fillna(df['bedrooms'])
    df['space_to_people_ratio'] = (df['bedrooms'] + df['bathrooms']) / df['accommodates']
    df['host_is_superhost'] = df['host_is_superhost'].fillna(1)
    return df

def select_features(df):    
    df_copy = df.copy()
    
    y = df_copy['guest_satisfaction']
    
    categorical_features = []
    numerical_features = []
    
    for col in df_copy.columns:
        if col == 'guest_satisfaction' or col == 'amenities_tokenized' or col == 'amenities':
            continue
        try:
            if df_copy[col].nunique() < 20:
                categorical_features.append(col)
            else: 
                numerical_features.append(col)
        except TypeError:
            continue
    numerical_features.append('property_type')
    numerical_features.append('neighbourhood_cleansed')
    numerical_features.append('zipcode')
    numerical_features.append('host_neighbourhood')
    numerical_features.append('host_location')
    numerical_features.append('street')
    
    print(f"Identified {len(categorical_features)} categorical features and {len(numerical_features)} numerical features")
    
    anova_results = []
    for col in numerical_features:
        groups = [group[col].dropna().values 
                 for _, group in df_copy.groupby('guest_satisfaction') if len(group) > 1]

        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            if not all(np.all(g == g[0]) for g in groups if len(g) > 0):
                try:
                    f_val, p_val = f_oneway(*groups)
                    anova_results.append({'feature': col, 'f_statistic': f_val, 'p_value': p_val})
                except Exception as e:
                    print(f"ANOVA failed for {col}: {e}")
    
    anova_df = pd.DataFrame(anova_results).sort_values(by='f_statistic', ascending=False)
    top_anova_features = anova_df.nlargest(5, 'f_statistic')['feature'].tolist() if not anova_df.empty else []
    print("Top ANOVA features:", top_anova_features)
    
    chi2_results = []
    for col in categorical_features:
        try:
            contingency = pd.crosstab(df_copy[col], df_copy['guest_satisfaction'])
            
            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                chi2, p, dof, expected = chi2_contingency(contingency)
                chi2_results.append({'feature': col, 'chi2': chi2, 'p_value': p})
        except Exception as e:
            print(f"Chi-squared failed for {col}: {e}")
    
    chi2_df = pd.DataFrame(chi2_results).sort_values(by='chi2', ascending=False)
    top_chi2_features = chi2_df.nlargest(5, 'chi2')['feature'].tolist() if not chi2_df.empty else []
    print("Top Chi-squared features:", top_chi2_features)
    
    X = df_copy.drop(columns=['guest_satisfaction', 'amenities_tokenized'])
    
    X_numeric = X.select_dtypes(include=['number'])
        
    try:
        np.random.seed(42)
        
        X_rf = X_numeric.copy()
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_rf, y)
        
        importances = rf.feature_importances_
        rf_results = pd.DataFrame({'feature': X_rf.columns, 'importance': importances})
        rf_results = rf_results.sort_values(by='importance', ascending=False)
        top_rf_features = rf_results.nlargest(5, 'importance')['feature'].tolist() if not rf_results.empty else []
        print("Top Random Forest features:", top_rf_features)
    except Exception as e:
        print(f"Random Forest feature importance calculation failed: {e}")
        top_rf_features = []
    
    all_top_features = []
    all_top_features.extend(top_anova_features)
    all_top_features.extend(top_chi2_features)
    all_top_features.extend(top_rf_features)
    
    final_features = []
    for feature in all_top_features:
        if feature not in final_features:
            final_features.append(feature)
    
    selected_features = final_features[:15]
    print(f"Final selected features: {selected_features}")
    
    return selected_features

def train_and_evaluate_model(model, param_grid, X_train, X_test, y_train, y_test, model_name, scaler=None):
    """Train and evaluate a model using GridSearchCV."""
    feature_names = X_train.columns.tolist()
    
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=2)
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    grid_search.fit(X_train, y_train)
    
    print(f"Best Hyperparameters for {model_name}: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    
    with open(f'best_{model_name}_model_Classification.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    y_pred = best_model.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    
    print("model accuracy : ",accuracy)
    return best_model, feature_names

def visualize_data(df, top_discrete_features, top_continuous_features):
    """Visualize data distributions and relationships."""
    rating_counts = df['guest_satisfaction'].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=rating_counts.index, y=rating_counts.values, color='skyblue')
    plt.title('Count of Each guest_satisfaction')
    plt.xlabel('guest_satisfaction Score (1 - 3)')
    plt.ylabel('Number of Listings')
    plt.xticks(rotation=90)
    plt.grid(True, axis='y')
    plt.show()
    
    numeric_df = df.select_dtypes(include=['number'])
    non_discrete_df = numeric_df.loc[:, ~numeric_df.columns.isin(top_discrete_features)]
    plt.figure(figsize=(40, 20))
    correlation_matrix = non_discrete_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Heatmap (Excluding Binary Columns)")
    plt.show()
    
    plt.figure(figsize=(20, 25))
    cols = 4
    for i, col in enumerate(top_continuous_features):
        plt.subplot((len(top_continuous_features) + cols - 1) // cols, cols, i + 1)
        sns.boxplot(y=df[col], color='skyblue')
        plt.title(f"{col} - Box Plot")
        plt.tight_layout()
    plt.show()

def main():
    """Main function to orchestrate the pipeline."""
    # Load and preprocess data
    df = load_data()
    df, binary_cols = preprocess_binary_columns(df)
    df = preprocess_numeric_columns(df)
    df = group_rare_values(df, 'zipcode')
    df, le_dict = preprocess_categorical_columns(df)
    df, imputer = impute_missing_values(df)
    df, date_cols = preprocess_dates(df)
    df, bad_binary_cols = drop_uninformative_columns(df)
    df, tfidf_dict, tfidf_scaler, pca, pca_columns = preprocess_text_columns(df)
    df, amenity_weights, amenity_scaler = compute_amenity_scores(df)
    df = feature_engineering(df)
    
    # Save preprocessed data
    df.to_csv('airbnb_preprocessed_tfidf_pca_classification.csv', index=False)
    for name in df['zipcode']:
        print(name)
    ##df = pd.read_csv('airbnb_preprocessed_tfidf_pca_classification.csv', low_memory=False)
    print(df.columns)
    
    # Save preprocessing objects
    preprocessing_artifacts = {
        'binary_cols': binary_cols,
        'le_dict': le_dict,
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
    
    with open('chosen_features.pkl', 'wb') as f:
        pickle.dump(chosen_features, f)
    
    model_df = df[[c for c in chosen_features if c in df.columns] + ['guest_satisfaction']].copy()
    model_df[chosen_features[:4]] = model_df[chosen_features[:4]].astype(int)
    
    X = model_df.drop(columns=['guest_satisfaction'])
    y = model_df['guest_satisfaction']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Define models and hyperparameters
    models = [
        (
            RandomForestClassifier(random_state=42),
            {
                'n_estimators': [100, 300, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5]
            },
            "random_forest",
            None
        ),

        (
            xgb.XGBClassifier(objective='accuracy', random_state=42),
            {
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [4, 6, 8],
                'n_estimators': [100, 300, 500]
            },
            "xgboost",
            None
        ),
        (
            CatBoostClassifier(cat_features=chosen_features[:4], silent=True, random_state=42),
            {
                'iterations': [1000],
                'learning_rate': [0.1],
                'depth': [10],
                'early_stopping_rounds' : [50]
                
            },
            "catboost"
            ,None
        ),
        (
            LogisticRegression(multi_class='multinomial', solver='saga', max_iter=1000),
            {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'tol': [1e-4, 1e-3]
            },
            "logistic_regression",
            StandardScaler()
        )
    ]
    for model, param_grid, model_name , model_scaler in models:
        best_model, feature_names = train_and_evaluate_model(model, param_grid, X_train, X_test, y_train, y_test, model_name, None)
        print(f"Saved model {model_name} with features: {feature_names}")
    
    visualize_data(df, chosen_features[:4], chosen_features[4:])

if __name__ == "__main__":
    main()