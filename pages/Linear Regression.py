import streamlit as st
from PIL import Image
st.markdown("# Linear Regression")
st.sidebar.markdown('''
# Performance
## `Selected features and scaling`
    - MAE: 2879.521
    - RMSE: 3756.43
    - R2 Score: 0.83
''')

image = Image.open('LinearRegression.jpg')

st.image(image, caption='Test data vs Prediction')

st.warning('''
### This is a decent result from the Linear Regression with feature selection and scaling
''')