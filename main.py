import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from data import generate_csv
import os
import pickle
import json

# 1️⃣ Carregar CSV
if os.path.exists('all_features.csv'):
    df = pd.read_csv('all_features.csv')
else:
    generate_csv()
    df = pd.read_csv('all_features.csv')
    
# 2️⃣ Separar features e labels
X = df.drop('label', axis=1).values.astype(np.float32)
y = df['label'].values.astype(np.int32)

# 3️⃣ Dividir treino/teste/validação
# Supondo que você queira usar 20% para teste e 20% do restante para validação
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

# 4️⃣ Criar modelo XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# 5️⃣ Treinar com early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# 6️⃣ Avaliar no conjunto de teste
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7️⃣ Salvar o modelo treinado
print("\n💾 Salvando modelo...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Modelo salvo em 'model.pkl'")

# 8️⃣ Salvar mapa de labels
label_map = {0: 'fake', 1: 'real'}
with open('label_map.json', 'w') as f:
    json.dump(label_map, f, indent=2)
print("✓ Label map salvo em 'label_map.json'")

print("\n✅ Treinamento concluído!")
print("Para fazer inferência com a pipeline paralelizada, use:")
print("  python inference.py <imagem_ou_diretório>")
