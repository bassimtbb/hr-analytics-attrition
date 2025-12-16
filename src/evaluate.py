import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib

# Load data
df = pd.read_csv("../data/raw/hr_attrition.csv")

X = df.drop("Attrition", axis=1)
y = df["Attrition"].map({"Yes": 1, "No": 0})

# Load trained model (Random Forest)
model = joblib.load("../models/Random_Forest.joblib")

# Predictions
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

# ------------------------
# 1. Confusion Matrix
# ------------------------
cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ------------------------
# 2. ROC Curve
# ------------------------
fpr, tpr, _ = roc_curve(y, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
# ------------------------
# 3. Feature Importance
# ------------------------
rf = model.named_steps["model"]
feature_names = model.named_steps["preprocessing"].get_feature_names_out()

importances = rf.feature_importances_

feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False).head(10)

plt.figure(figsize=(6,4))
sns.barplot(x="importance", y="feature", data=feat_imp)
plt.title("Top 10 Feature Importances")
plt.show()
