import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline

# Load dataset
file_path = 'new_vega_rotation_overshooting_alpha_mlt_test_no_num_BIGGER_num_gt_180.csv'
data = pd.read_csv(file_path, delimiter=",", header=0)

# Define features and target
x = data[['mass', 'z', 'fov', 'mlt', 'age', 'teff', 'lum']]
y = data['log_k']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Regressor
model = DecisionTreeRegressor(random_state=123)

# Initialize RFE with the model
rfe = RFE(estimator=model, n_features_to_select=7)

# Create a pipeline with RFE and the model
pipeline = Pipeline(steps=[('s', rfe), ('m', model)])

# Fit the pipeline on the training data
pipeline.fit(x_train, y_train)

# Make predictions on the feature set only
yhat = pipeline.predict(x)

# Print predictions
print('Predicted:', yhat)


# Extract the seventh prediction
seventh_prediction = yhat[6]

# Extract the corresponding row of features
seventh_features = x.iloc[6]

# Print the features and the corresponding prediction
print("Features:")
print(seventh_features)
print("\nPredicted log_k: %.3f" % seventh_prediction)