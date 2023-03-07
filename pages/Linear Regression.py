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

with open("LinearRegression.jpg", "rb") as file:
    btn = st.download_button(
            label="Download image",
            data=file,
            file_name="LinearRegression.jpg",
            mime="image/jpg",
            
            use_container_width=True
          )


st.warning('''
### This is a decent result from the Linear Regression with feature selection and scaling
''')

st.markdown('### Compared with Random Forest Regression')
col1, col2, col3 = st.columns(3)
col1.metric("MAE", "2879.521", "+427.641")
col2.metric("RMSE", "3756.43", "+175.9")
col3.metric("R2 Score", "0.83", "-0.01")

st.info('MAE and RMSE should be lower for high performance & R2 score should be greater',icon='ℹ️')
