import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
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

print("\n📊 Distribuição do dataset:")
print(f"   Total de amostras: {len(y)}")
print(f"   Fake (0): {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
print(f"   Real (1): {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")

# 3️⃣ Dividir treino/teste/validação
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

# 4️⃣ Normalização dos dados (importante para reduzir viés)
print("\n🔧 Normalizando features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 5️⃣ Balanceamento com SMOTE (se necessário)
print("\n⚖️  Verificando balanceamento...")
train_fake = sum(y_train == 0)
train_real = sum(y_train == 1)
print(f"   Treino - Fake: {train_fake}, Real: {train_real}")

if abs(train_fake - train_real) / len(y_train) > 0.1:  # Desbalanceado > 10%
    print("   Aplicando SMOTE para balanceamento...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"   Após SMOTE - Fake: {sum(y_train == 0)}, Real: {sum(y_train == 1)}")
else:
    print("   Dataset já está balanceado ✓")

# 6️⃣ Calcular scale_pos_weight para lidar com desbalanceamento
scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
print(f"\n📏 Scale pos weight: {scale_pos_weight:.2f}")

# 7️⃣ Criar modelo XGBoost melhorado
print("\n🤖 Criando modelo XGBoost otimizado...")
model = XGBClassifier(
    n_estimators=300,           # Mais árvores
    max_depth=4,                # Menos profundidade (evita overfitting)
    learning_rate=0.05,         # Learning rate menor (mais conservador)
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,  # Ajusta para desbalanceamento
    min_child_weight=3,         # Evita overfitting
    gamma=0.1,                  # Regularização
    reg_alpha=0.1,              # L1 regularization
    reg_lambda=1.0,             # L2 regularization
    early_stopping_rounds=20,   # Early stopping integrado
    random_state=42,
    eval_metric=['logloss', 'auc']
)

# 8️⃣ Treinar com early stopping
print("\n🚀 Treinando modelo...")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50  # Mostra progresso a cada 50 iterações
)

# 9️⃣ Avaliar no conjunto de teste
print("\n📊 Avaliando modelo...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Métricas
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba[:, 1])

print("\n" + "="*80)
print("RESULTADOS DA AVALIAÇÃO")
print("="*80)
print(f"\n✓ Accuracy: {acc:.4f}")
print(f"✓ ROC-AUC: {auc:.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

print("\n🎯 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                Predito")
print(f"              Fake  Real")
print(f"Real  Fake    {cm[0][0]:4d}  {cm[0][1]:4d}")
print(f"      Real    {cm[1][0]:4d}  {cm[1][1]:4d}")

# Análise de viés
fake_recall = cm[0][0] / (cm[0][0] + cm[0][1])
real_recall = cm[1][1] / (cm[1][0] + cm[1][1])
print(f"\n📈 Análise de Viés:")
print(f"   Recall Fake: {fake_recall:.2%}")
print(f"   Recall Real: {real_recall:.2%}")
print(f"   Diferença: {abs(fake_recall - real_recall):.2%}")

if abs(fake_recall - real_recall) < 0.1:
    print("   ✅ Modelo bem balanceado!")
elif fake_recall > real_recall:
    print("   ⚠️  Modelo tendencioso para FAKE")
else:
    print("   ⚠️  Modelo tendencioso para REAL")

# 🔟 Salvar o modelo e o scaler
print("\n💾 Salvando modelo e preprocessamento...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Modelo salvo em 'model.pkl'")

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler salvo em 'scaler.pkl'")

# 1️⃣1️⃣ Salvar mapa de labels e métricas
label_map = {0: 'fake', 1: 'real'}
with open('label_map.json', 'w') as f:
    json.dump(label_map, f, indent=2)
print("✓ Label map salvo em 'label_map.json'")

metrics = {
    'accuracy': float(acc),
    'roc_auc': float(auc),
    'fake_recall': float(fake_recall),
    'real_recall': float(real_recall),
    'n_train': len(y_train),
    'n_test': len(y_test)
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("✓ Métricas salvas em 'metrics.json'")

print("\n" + "="*80)
print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
print("="*80)
print("\n📝 Próximos passos:")
print("   1. python inference.py <imagem> - Classificar imagens")
print("   2. streamlit run app.py - Interface web")
print("   3. python visualize_pipeline.py - Visualização interativa")
print()
