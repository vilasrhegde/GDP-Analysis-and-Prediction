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

st.success('''
### Gradient Boosting gave us pretty good performance overall, that too without the need of optimisation!
> Although RandomForest and GradientBoosting are comparable with respect to our same dataset.
''')