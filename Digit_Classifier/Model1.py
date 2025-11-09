from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1) Load 8x8 grayscale digits (0–9)
digits = load_digits()
X, y = digits.data, digits.target

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Scale features (helps linear models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 4) Train a simple classifier (Logistic Regression)
clf = LogisticRegression(max_iter=2000, n_jobs=None)
clf.fit(X_train, y_train)

# 5) Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6) (Optional) see a few predictions
fig, axes = plt.subplots(1, 6, figsize=(9, 2))
for ax, img, true, pred in zip(axes, digits.images[:6], y[:6], clf.predict(scaler.transform(digits.data[:6]))):
    ax.imshow(img, cmap="gray_r")
    ax.set_title(f"T:{true} P:{pred}")
    ax.axis("off")
plt.tight_layout()
plt.show()
