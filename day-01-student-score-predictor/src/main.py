import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('data/student_scores.csv')

print(df)

X = df[['hours_studied']]
y = df['score']

X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size= 0.2,
  random_state= 42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

print("Mean Absolute Error:", mae)

prediction = model.predict([[7.5]])
print("Prediction score:", prediction[0]);

plt.scatter(X, y, label="Actual scores")

plt.plot(
    X,
    model.predict(X),
    label="Regression line"
)

plt.xlabel("Hours studied")
plt.ylabel("Score")
plt.title("Student Score Prediction")

plt.legend()

plt.savefig("regression_plot.png")

plt.show()