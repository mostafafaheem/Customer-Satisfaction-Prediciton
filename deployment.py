import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

@st.cache_resource
def load_models():
    rf_model = load('best_random_forest_model.pkl')
    cat_model = load('best_catboost_model.pkl')
    xg_model = load('best_xgboost_model.pkl')
    poly_model = load('best_polynomial_model.pkl')
    svr_model = load('best_svr_model.pkl')
    
    return rf_model, cat_model, xg_model, poly_model, svr_model

# Fixed model assignment order to match return statement
rf_model, cat_model, xg_model, poly_model, svr_model = load_models()

def make_predictions(input_data):
    rf_pred = rf_model.predict(input_data)[0]
    cat_pred = cat_model.predict(input_data)[0]
    xg_pred = xg_model.predict(input_data)[0]
    poly_pred = poly_model.predict(input_data)[0]
    svr_pred = svr_model.predict(input_data)[0]

    return rf_pred, cat_pred, xg_pred, poly_pred, svr_pred

def main():
    st.title("Airbnb Review Score Predictor")
    st.write("This app predicts the review scores rating based on Airbnb listing features.")
    
    st.sidebar.header('Input Listing Features')
    
    # Updated to include all features from chosen_features.pkl
    host_is_superhost = st.sidebar.selectbox('Is the host a superhost?', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    instant_bookable = st.sidebar.selectbox('Is instant booking available?', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    is_frequently_booked = st.sidebar.selectbox('Is the listing frequently booked?', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    is_host_verified = st.sidebar.selectbox('Is the host verified on Airbnb?',["Yes","No"])
    host_total_listings_count = st.sidebar.number_input('Total listings count of the host', min_value=1, value=10)
    cat_pca_1 = st.sidebar.slider('Categorical Features PCA Component',min_value=-5.0,max_value=5.0,value=0.0,step=0.1)
    tfidf_pca_1 = st.sidebar.slider('TF-IDF PCA Component 1', min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    amenity_score = st.sidebar.number_input('Amenity score (0-100)', min_value=0.0, max_value=100.0, value=50.0)
    
    actual_rating = st.sidebar.number_input('Actual review score (optional)', min_value=0.0, max_value=100.0, value=None)
    
    if st.sidebar.button('Predict Review Score'):
        # Updated to use features from chosen_features.pkl
        input_data = pd.DataFrame({
            'host_is_superhost': [host_is_superhost],
            'instant_bookable': [instant_bookable],
            'is_frequently_booked': [is_frequently_booked],
            'host_identity_verified':[is_host_verified],
            'host_total_listings_count': [host_total_listings_count],
            'cat_pca_1': [cat_pca_1],
            'tfidf_pca_1': [tfidf_pca_1],
            'amenity_score': [amenity_score]
        })
        
        
        
        rf_pred, cat_pred, xg_pred, poly_pred, svr_pred = make_predictions(input_data)
        
        st.subheader("Prediction Results")
        results = pd.DataFrame({
            'Model': ['Random Forest', 'CatBoost', 'XGBoost', 'Polynomial Regression', 'SVR'],
            'Predicted Score': [rf_pred, cat_pred, xg_pred, poly_pred, svr_pred]
        })
        
        if actual_rating is not None:
            results['Actual Score'] = actual_rating
            results['Difference'] = results['Predicted Score'] - results['Actual Score']
        
        st.dataframe(results.style.format({
            'Predicted Score': '{:.1f}',
            'Actual Score': '{:.1f}',
            'Difference': '{:.1f}'
        }), use_container_width=True)
        
        if actual_rating is not None:
            st.subheader("Prediction Accuracy")
            # Fixed to use 5 columns for 5 models
            cols = st.columns(5)
            with cols[0]:
                delta = rf_pred - actual_rating
                st.metric("Random Forest", f"{rf_pred:.1f}", delta=f"{delta:.1f}")
            with cols[1]:
                delta = cat_pred - actual_rating
                st.metric("CatBoost", f"{cat_pred:.1f}", delta=f"{delta:.1f}")
            with cols[2]:
                delta = xg_pred - actual_rating
                st.metric("XGBoost", f"{xg_pred:.1f}", delta=f"{delta:.1f}")
            with cols[3]:
                delta = poly_pred - actual_rating
                st.metric("Polynomial", f"{poly_pred:.1f}", delta=f"{delta:.1f}")
            with cols[4]:
                delta = svr_pred - actual_rating
                # Fixed typo: set.metric -> st.metric
                st.metric("SVR", f"{svr_pred:.1f}", delta=f"{delta:.1f}")
        
        st.subheader("Input Features")
        st.dataframe(input_data)
        
        st.write("**Note:** The review score rating is predicted on a scale from 0 to 100.")

if __name__ == '__main__':
    main()