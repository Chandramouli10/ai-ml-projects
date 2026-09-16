import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

iris = load_iris()

X = iris.data
y = iris.target

print("Features:")
print(X)

print("Target:")
print(y)

print("X Shape:", X.shape)
print('Y Shape:', y.shape)

X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size= 0.2,
  random_state= 42
)

print("Training features", X_train.shape)
print("Testing features", X_test.shape)
print("Training targets", y_train.shape)
print("Testing targets", y_test.shape)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Prediction Labels:", y_pred)
print("Actual Labels:", y_test)

accuracy = accuracy_score(y_pred, y_test)

print("Accuracy:", accuracy)
print("Accuracy percentage", accuracy * 100, "%")

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

display.plot()
plt.title("Iris Flower Confusion Matrix")
plt.show()

print("Classification Reports")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

custom_flowers = [
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 2.9, 4.5, 1.5],
    [6.5, 3.0, 5.5, 1.8]
]

predictions = model.predict(custom_flowers)

for prediction in predictions:
  flower_name = iris.target_names[prediction] 
  print(flower_name)