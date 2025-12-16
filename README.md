# HR Analytics – Attrition Prediction (Machine Learning + Streamlit)

## 🎯 Objectif
Construire un modèle de Machine Learning capable de prédire le risque de départ des employés (**attrition**) à partir de données RH, puis déployer le modèle dans une application **Streamlit** pour une utilisation simple par des profils non techniques.

## 📊 Dataset
- Dataset RH (attrition binaire : `Yes/No`)
- Taille : **1470 lignes / 35 colonnes**
- Variable cible : **Attrition**

## 🧠 Approche
1. **EDA (Exploratory Data Analysis)**  
   - Vérification distribution de la cible (déséquilibre de classes)
   - Analyse de variables clés (Age, MonthlyIncome, JobSatisfaction, …)
2. **Prétraitement**
   - Séparation variables **numériques** et **catégorielles**
   - `StandardScaler` pour les numériques
   - `OneHotEncoder` pour les catégorielles
   - Pipeline complet (évite la fuite de données)
3. **Modélisation**
   - **Logistic Regression** (baseline interprétable)
   - **Random Forest** (modèle plus robuste)
4. **Évaluation**
   - ROC-AUC
   - Matrice de confusion
   - Courbe ROC
5. **Interprétation**
   - Feature importance (Random Forest)  
   - Variables souvent influentes : **MonthlyIncome**, **Age**
6. **Déploiement**
   - Application **Streamlit** : upload CSV → prédictions + probabilité

## ⚙️ Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit
- Joblib

## 📁 Structure du projet
```text
hr-analytics-attrition/
├── data/
│   └── raw/
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── train.py
│   └── evaluate.py
├── models/
├── app.py
├── requirements.txt
└── README.md
```
▶️ Installation
bash
```text
py -m pip install -r requirements.txt
```
🏋️ Entraîner les modèles
bash
```text
python src/train.py
```
📈 Évaluer le modèle
bash
```text
python src/evaluate.py
```
🚀 Lancer l’application Streamlit
bash
```text
py -m pip install streamlit
streamlit run app.py
```
📌 Auteur
Bassim Tabbeb

GitHub : https://github.com/bassimtbb

LinkedIn : https://linkedin.com/in/tabbeb-bassim
