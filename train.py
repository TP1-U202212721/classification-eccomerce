from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

dataset = pd.read_csv('dataset/dataset_ventas.csv')
features = dataset.drop('etiqueta_riesgo', axis=1)
labels = dataset['etiqueta_riesgo']
X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels, 
    test_size=0.2, 
    random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
joblib.dump(model, 'train/model/estimation_risk_model.pkl')
print(classification_report(y_test, y_pred))