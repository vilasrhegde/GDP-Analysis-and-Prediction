import streamlit as st
from PIL import Image
st.markdown("# Gradient Boosting")
st.sidebar.markdown('''
# Performance
## `Selected features and scaling`
    - MAE: 2467.21
    - RMSE: 3789.30
    - R2 Score: 0.83
''')

image = Image.open('GradientBoosting.jpg')

st.image(image, caption='Test data vs Prediction')
with open("GradientBoosting.jpg", "rb") as file:
    btn = st.download_button(
            label="Download image",
            data=file,
            file_name="GradientBoosting.jpg",
            mime="image/jpg",
            
            use_container_width=True
          )

st.success('''
### Gradient Boosting gave us pretty good performance overall, that too without the need of optimisation!
> Although RandomForest and GradientBoosting are comparable with respect to our same dataset.
''')

st.markdown('### Compared with Random Forest Regression')
col1, col2, col3 = st.columns(3)
col1.metric("MAE", "2467.21", "+15.33")
col2.metric("RMSE", "3789.30", "+208.77")
col3.metric("R2 Score", "0.83", "-0.01")

st.info('MAE and RMSE should be lower for high performance & R2 score should be greater',icon='ℹ️')

