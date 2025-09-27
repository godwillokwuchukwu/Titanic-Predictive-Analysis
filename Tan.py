import streamlit as st
import joblib

# Dummy credentials (replace with a database check if needed)
USERNAME = "admin"
PASSWORD = "password123"

# Function for authentication
def authenticate(username, password):
    return username == USERNAME and password == PASSWORD

# Function to log out and return to login interface
def logout():
    st.session_state.logged_in = False
    st.rerun()  # Redirects back to login interface

# Initialize session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login interface
if not st.session_state.logged_in:
    st.title("Login to Titanic Prediction")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.logged_in = True
            st.success("Login successful! Redirecting...")
            st.rerun()  # Refresh the page after login
        else:
            st.error("Invalid username or password")

# If logged in, show the Titanic prediction interface
if st.session_state.logged_in:
    # Load the model
    model = joblib.load('Titanic_model_lr.model')

    # Set up the UI
    st.title("Titanic Survival Prediction")
    st.markdown("Enter passenger details below to predict survival:")

    # Logout button (Calls the logout function)
    if st.button("Logout"):
        logout()  # Redirects back to login page

    # Create columns for better layout
    col1, col2 = st.columns(2)

    # Input fields
    Pclass = col1.number_input("Passenger Class (1st, 2nd, 3rd)", min_value=1, max_value=3)
    Sex = col2.radio("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
    Age = col1.number_input("Age", min_value=0, max_value=80)
    Siblings_spouses = col2.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10)
    parent_child = col1.number_input("Parents/Children Aboard", min_value=0, max_value=6)
    Fare = col2.number_input("Fare", min_value=0.0, max_value=500.0, step=0.1)

    # Prediction button
    if st.button("Predict"):
        input_data = [[Pclass, Sex, Age, Siblings_spouses, parent_child, Fare]]
        prediction = model.predict(input_data)

        # Display result
        if prediction[0] == 1:
            st.success("The model predicts that the passenger **survived**.")
        else:
            st.error("The model predicts that the passenger **did not survive**.")
