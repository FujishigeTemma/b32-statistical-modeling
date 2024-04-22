import polars as pl
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pl.read_csv("data/penguins.csv")
columns = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
selected_species = ["Adelie", "Chinstrap"]

df = df.filter(pl.col("species").is_in(selected_species))
df = df.drop_nulls()

X = df.select(columns).to_numpy()
y = df.select("species").to_series().to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train)

y_pred = lda.predict(X_test_scaled)

accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy:.3f}")
