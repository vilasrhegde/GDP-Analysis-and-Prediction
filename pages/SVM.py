import streamlit as st
from PIL import Image
st.markdown("# SVM")
st.sidebar.markdown('''
# Performance
## `Selected features and scaling`
    - MAE: 7040.04
    - RMSE: 9794.59
    - R2 Score: -0.16
''')

image = Image.open('SVM.jpg')

st.image(image, caption='Test data vs Prediction')

with open("SVM.jpg", "rb") as file:
    btn = st.download_button(
            label="Download image",
            data=file,
            file_name="SVM.jpg",
            mime="image/jpg",
            
            use_container_width=True
          )


st.error('''
### Feature scaling and selection did not help much for the prediction in our case, hence results of SVM is worse than Linear Regression. 
''')

st.markdown('### Compared with Random Forest Regression')
col1, col2, col3 = st.columns(3)
col1.metric("MAE", "7040.04", "-15.33")
col2.metric("RMSE", "9794.59", "0.01")
col3.metric("R2 Score", "-0.16", "-1")

st.info('MAE and RMSE should be lower for high performance & R2 score should be greater',icon='ℹ️')
