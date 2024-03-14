# Cel: przewidywanie ceny (kolumna MEDV) w zależności  od innych cech.
# MEDV - mediana wartości domów z danego terenu (w tys. dolarów)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#import seaborn as sns
from sklearn.metrics import r2_score

columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
           'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

data = pd.read_csv(r"C:\Users\cayna\Desktop\housing.data", sep=' +', engine='python', header=None,
                   names=columns)

columns_to_X = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# sns.pairplot(data[columns], height=1.5) # Wykres korelacji - UWAGA długo się ładuje - odkomentuj 'import seaborn as sns'
# Najsilniejsze korelacje: ZN, INDUS, NOX, RM, AGE, DIS, PTRATIO, LSTAT
other_columns = ['ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'PTRATIO', 'LSTAT']


def evaluate_models(X, y):
    results = []
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        score = r2_score(y_test, y_pred)

        results.append(score)

    return np.mean(results), np.std(results)


# Ocena modelu z pełnym zestawem cech
full_features_mean, full_features_std = evaluate_models(data[columns_to_X], data['MEDV'])

# Ocena modelu z "lepszym" zestawem cech
better_features_mean, better_features_std = evaluate_models(data[other_columns], data['MEDV'])

print(f'Pełny zestaw cech - Średni R^2: {full_features_mean:.4f}, Std: {full_features_std:.4f}')
print(f'Wybrany zestaw cech - Średni R^2: {better_features_mean:.4f}, Std: {better_features_std:.4f}')


#Przykładowy wiersz danych |['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS','RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
example = np.array([[0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296.0, 15.3, 396.9, 4.98]])
example_df = pd.DataFrame(example, columns=columns_to_X) #Przekształcenie na data frame

X = data[columns_to_X]
y = data['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

example_scaled = scaler.transform(example_df)

predicted_price = lr.predict(example_scaled)

print(f"Przewidywana cena: {predicted_price[0]}")