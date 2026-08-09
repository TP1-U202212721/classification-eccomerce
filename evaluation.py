import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

label_name_map = {1: "High Risk", 2: "Moderate Risk", 3: "Low Risk"}

dataset = pd.read_csv('dataset/dataset_ventas.csv')
features = dataset.drop('etiqueta_riesgo', axis=1)
labels = dataset['etiqueta_riesgo']
X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels, 
    test_size=0.2, 
    random_state=42
)

model = joblib.load('model/estimation_risk_model.pkl')
y_pred = model.predict(X_test)

class_values = np.unique(np.concatenate((y_test.values, y_pred)))
harvest = confusion_matrix(y_test, y_pred, labels=class_values)
metrics = classification_report(y_test, y_pred, output_dict=True)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(harvest, interpolation='nearest', cmap=plt.cm.Blues)

class_names = [label_name_map.get(int(v), str(v)) for v in class_values]
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')

thresh = harvest.max() / 2.0 if harvest.size else 0
for i in range(harvest.shape[0]):
    for j in range(harvest.shape[1]):
        color = "white" if harvest[i, j] > thresh else "black"
        ax.text(j, i, format(harvest[i, j], 'd'), ha="center", va="center", color=color)

ax.set_title("Confusion Matrix")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.tight_layout()
plt.savefig('results/confusion_matrix.png')


df = pd.DataFrame(metrics).transpose()
df.loc[[k for k in df.index if k not in ("macro avg","weighted avg", "accuracy")], ['recall','f1-score', 'precision']].plot.bar(rot=45)
plt.ylabel('Score')
plt.title('Classification Report Metrics')
plt.tight_layout()
plt.savefig('results/classification_report_metrics.png')



