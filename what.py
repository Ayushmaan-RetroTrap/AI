import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

file_path = "Color_Combinations.csv"
data = pd.read_csv(file_path)

data.columns = data.columns.str.strip().str.lower()

input_col = 'color 1 name'
match_cols = ['color 2 name']

input_encoder = OneHotEncoder(sparse_output=False)

X = input_encoder.fit_transform(data[[input_col]])
y = data[match_cols].values.ravel()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

def predict_matches(user_input):
    input_array = input_encoder.transform([[user_input]])
    predicted_probs = model.predict_proba(input_array)
    matches = []
    for i, prob in enumerate(predicted_probs[0]):
        if prob > 0:
            matches.append((model.classes_[i], prob * 100))
    matches.sort(key=lambda x: x[1], reverse=True)
    return [f"Match: {match[0]} with probability {match[1]:.2f}%" for match in matches]

user_input = input("Enter a color: ").strip()
try:
    print(f"All predicted combinations for '{user_input}':")
    for match in predict_matches(user_input):
        print(match)
except Exception as e:
    print(f"Error: {e}. Make sure the input is valid.")