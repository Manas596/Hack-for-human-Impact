import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load data (update path if needed)
df = pd.read_csv("2025-11-15T16-32_export.csv")

# 2. Binary Y/N → 0/1
binary_cols = ['OPENSPACE', 'HOUSING', 'RECREATION', 'HISTORIC', 'LANDBUY', 'HAS_GEOG']

for col in binary_cols:
    df[col] = df[col].map({'Y': 1, 'N': 0})

# 3. Compute description length
df['DESCR_LEN'] = df['DESCR'].astype(str).str.len()

# 4. Encode STATUS -> 0/1 (target)
df['STATUS_BIN'] = df['STATUS'].map({
    'Project complete': 1,
    'Project in progress': 0,
    'Project cancelled': 0
})

# Drop rows where mapping failed
df = df.dropna(subset=['STATUS_BIN'])

# 5. One-hot encode TOWN
df = pd.get_dummies(df, columns=['TOWN'], drop_first=True)

# 6. Build feature list
base_features = [
    'APPR_YR', 'OPENSPACE', 'HOUSING', 'RECREATION', 'HISTORIC',
    'CPA_TOT', 'TOT_COST', 'HOUSTOT', 'LANDBUY', 'HAS_GEOG', 'DESCR_LEN'
]

# Include one-hot town columns
town_features = [c for c in df.columns if c.startswith('TOWN_')]
features = base_features + town_features

X = df[features]
y = df['STATUS_BIN']

print("Shape of X, y:", X.shape, y.shape)  # <-- should NOT be (0, something)

# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 8. Scale numeric columns
numeric_cols = ['APPR_YR', 'CPA_TOT', 'TOT_COST', 'HOUSTOT', 'DESCR_LEN']

scaler = StandardScaler()
X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols])

# 9. Fit logistic regression
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# 10. Evaluate
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 11. Feature importance
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("\nTop features:\n", coef_df.head(15))
