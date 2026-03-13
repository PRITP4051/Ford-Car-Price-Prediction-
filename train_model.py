import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load data
df = pd.read_csv("data/ford.csv")

# Label encode model column
le = LabelEncoder()
df["model"] = le.fit_transform(df["model"])

# One hot encoding
df = pd.get_dummies(df, columns=["transmission","fuelType"], drop_first=True)

# Features
X = df.drop("price", axis=1)
y = df["price"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

results = []

def evaluate(model,name,Xtr,Xte):

    model.fit(Xtr,y_train)

    train_pred = model.predict(Xtr)
    test_pred = model.predict(Xte)

    results.append({
        "Model":name,
        "Train_R2":r2_score(y_train,train_pred),
        "Test_R2":r2_score(y_test,test_pred),
        "RMSE":np.sqrt(mean_squared_error(y_test,test_pred)),
        "MAE":mean_absolute_error(y_test,test_pred)
    })

# Linear Regression
evaluate(LinearRegression(),"Linear",X_train,X_test)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

evaluate(LinearRegression(),"Polynomial",X_train_poly,X_test_poly)

# Ridge
evaluate(Ridge(),"Ridge",X_train,X_test)

# Lasso
evaluate(Lasso(),"Lasso",X_train,X_test)

# Decision Tree
evaluate(DecisionTreeRegressor(),"DecisionTree",X_train,X_test)

scores = pd.DataFrame(results)

scores.to_csv("reports/model_scores.csv",index=False)

# Train final polynomial model
final_model = LinearRegression()
final_model.fit(X_train_poly,y_train)

# Save models
pickle.dump(final_model,open("models/polynomial_model.pkl","wb"))
pickle.dump(poly,open("models/poly_transformer.pkl","wb"))
pickle.dump(le,open("models/label_encoder.pkl","wb"))

trans_cols = [c for c in X.columns if "transmission_" in c]
fuel_cols = [c for c in X.columns if "fuelType_" in c]

pickle.dump(trans_cols,open("models/transmission_cols.pkl","wb"))
pickle.dump(fuel_cols,open("models/fuel_cols.pkl","wb"))

print("Training complete.")