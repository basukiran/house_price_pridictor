# House Price Predictor

A Streamlit app that estimates an Ames home sale price from a few basic details. The form uses sensible defaults for optional details and shows a rough range based on the model's test RMSE.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads the trained model from `housing_model.pkl`.
