import streamlit as st
from PIL import Image
st.markdown("# Gradient Boosting")
st.sidebar.markdown('''
# Performance
## `Selected features and scaling`
    - MAE: 7040.04
    - RMSE: 9794.59
    - R2 Score: -0.16
''')

image = Image.open('SVM.jpg')

st.image(image, caption='Test data vs Prediction')

st.error('''
### Feature scaling and selection did not help much for the prediction in our case, hence results of SVM is worse than Linear Regression. 
''')