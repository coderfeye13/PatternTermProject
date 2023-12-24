import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Veriyi oku ve tipleri kontrol et (DtypeWarning önlemi)
csvPath = r"otu.csv"
dataset = pd.read_csv(csvPath, dtype=str)

# Veri setini düzenle
X = dataset.iloc[1:, :].T
y = dataset.iloc[:1, :].T.squeeze()
le = LabelEncoder()
y = le.fit_transform(y)

# Veriyi eğitim ve test setlerine böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Veriyi normalize et ve MLP modelini oluştur
model = make_pipeline(StandardScaler(), MLPClassifier(random_state=0, max_iter=300))

# Modeli eğit
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yap
y_pred = model.predict(X_test)

# Model performansını değerlendir
cMatrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cMatrix)

classificationReport = classification_report(y_test, y_pred)
print("Classification Report:\n", classificationReport)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy, "\n")

sensitivity = cMatrix[0, 0]/(cMatrix[0, 0]+cMatrix[0, 1])
print('Sensitivity : ', sensitivity)

specificity = cMatrix[1, 1]/(cMatrix[1, 0]+cMatrix[1, 1])
print('Specificity : ', specificity)

ROC_AUC = roc_auc_score(y_test, y_pred)
print('ROC AUC : {:.4f}'.format(ROC_AUC))