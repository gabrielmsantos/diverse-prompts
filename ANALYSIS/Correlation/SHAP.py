import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import shap

# Load data
MAPELITES_PATH = 'FEATURES/MAPELITES/PARQUET/'
df = pd.read_parquet(MAPELITES_PATH + 'extracted_features_starling-lm-7b-alpha-bii_winowhy.parquet')

# Calculate performance column
df['performance'] = (df['model_formatted_output'] == df['correct']).astype(int)

# Select numeric features
numeric_cols = ['num_examples', 'instance_word_count', 'prompt_word_count', 
               'num_steps', 'num_special_tokens', 'prompt_type_token_ratio',
               'includes_context']

# Prepare features
X = df[numeric_cols]
y = df['performance']

# Train random forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Calculate SHAP values
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

# Summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, plot_type="bar")
plt.tight_layout()
plt.show()

# Detailed summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X)
plt.tight_layout()
plt.show()

# Create dependence plots for each feature
for feature in numeric_cols:
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature, shap_values, X)
    plt.tight_layout()
    plt.show()

# Create interaction plots between pairs of important features
feature_pairs = [
    ('num_examples', 'num_steps'),
    ('num_examples', 'num_special_tokens'),
    ('num_steps', 'num_special_tokens')
]

for feat1, feat2 in feature_pairs:
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feat1, shap_values, X, interaction_index=feat2)
    plt.tight_layout()
    plt.show()
