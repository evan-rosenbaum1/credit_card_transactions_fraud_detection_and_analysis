import os
import pickle
import pandas as pd
import streamlit as st

# Load pre-trained model
def load_model(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

# Define function to create age_group
def create_age_group(age):
    if age < 10:
        return 'Gen Alpha'
    elif age < 26:
        return 'Gen Z'
    elif age < 42:
        return 'Millennial'
    elif age < 58:
        return 'Gen X'
    elif age < 77:
        return 'Baby Boomer'
    else:
        return 'Silent Generation'

def preprocess_data(df, user_data):
    # Copy Original DataFrame
    original_df = df.copy()

    # Reformat column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Add user data and create necessary columns
    df['age'] = user_data['age']
    df['state'] = user_data['state']
    df['job'] = user_data['job']
    df['age_group'] = user_data['age_group']
    df['day_of_week'] = pd.to_datetime(df['transaction_date']).dt.dayofweek
    df['month'] = pd.to_datetime(df['transaction_date']).dt.month

    # Select relevant columns for model input
    model_df = df[['category', 'amt', 'state', 'job', 'day_of_week', 'month', 'age', 'age_group']]

    return model_df, original_df

# Questionnaire Page
def questionnaire():
    st.title("Questionnaire")

    us_state_abbreviations = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 
        'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 
        'VA', 'WA', 'WV', 'WI', 'WY'
    ]
    
    state = st.selectbox('State', us_state_abbreviations)
    job = st.selectbox('Job', [
        'Engineering and Technology',
        'Medical and Healthcare',
        'Education and Research',
        'Arts, Media, and Design',
        'Business and Management',
        'Law and Public Service',
        'Science and Environment',
        'Transportation and Logistics',
        'Agriculture and Rural Affairs',
        'Construction and Architecture'
    ])
    age = st.number_input('Age', max_value=120, value=18)
    
    if st.button('Submit'):
        st.session_state['user_data'] = {
            'state': state,
            'job': job,
            'age': age,
            'age_group': create_age_group(age)
        }
        st.session_state['page'] = 'upload'

# CSV Upload Page
def csv_upload(model):
    st.title("Upload CSV")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        st.write("CSV file uploaded. Processing...")
        data_df = pd.read_csv(uploaded_file)

        # Process data with user inputs and model preprocessor
        processed_data, original_df = preprocess_data(data_df, st.session_state['user_data'])
        
        # Make predictions
        predictions = model.predict(processed_data)

        # Map predictions to labels
        prediction_labels = ['non-fraudulent' if p == 0 else 'potential_fraud' for p in predictions]

        # Add predictions to the original DataFrame
        original_df['predictions'] = prediction_labels

        # Save the updated DataFrame to a new CSV file
        csv_file_path = 'updated_predictions.csv'
        original_df.to_csv(csv_file_path, index=False)
        
        # Store both DataFrames and file path in session state
        st.session_state['csv_file_path'] = csv_file_path
        st.session_state['original_df'] = original_df
        st.write("CSV file processed and predictions added.")
        
        # Button to navigate to results page
        if st.button('Go to Results'):
            st.session_state['page'] = 'results'

# Results Page
def results():
    st.title("Results")
    
    st.write("Here is your file with predictions:")
    
    csv_file_path = st.session_state.get('csv_file_path')
    original_df = st.session_state.get('original_df')
    
    if csv_file_path and os.path.exists(csv_file_path):
        st.write("Preview of the updated data:")
        st.dataframe(original_df)  # Display the updated DataFrame

        # Provide download button
        with open(csv_file_path, 'rb') as file:
            st.download_button(
                label="Download CSV with Predictions",
                data=file,
                file_name=csv_file_path,
                mime="text/csv"
            )
    else:
        st.write("No file to display or file path is incorrect.")

# Main function
def main():
    if 'page' not in st.session_state:
        st.session_state['page'] = 'questionnaire'
    
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'feat_min_rf_model.pkl')
    model = load_model(model_path)

    # Display the correct page based on the state
    if st.session_state['page'] == 'questionnaire':
        questionnaire()
    elif st.session_state['page'] == 'upload':
        csv_upload(model)
    elif st.session_state['page'] == 'results':
        results()
    else:
        st.write("Page not found.")

if __name__ == "__main__":
    main()
