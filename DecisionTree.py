import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

df = pd.read_csv('Dataset/insur/insurance.csv')

print(df.head())
x = df[df.columns[0:8]]
print(x.head())
y = df[df.columns[-1]]
print(y.head())
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.5, test_size=0.5, shuffle = False)
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)
print(y_train.head())
print(x_test.iloc[0])
export_graphviz(
            model,
            out_file =  "Insurance.dot",
            feature_names = x_train.columns[0:8],
            class_names = True,
            filled = True,
            rounded = True)
model.predict(x_test)
print("Accuracy - ",model.score(x_test,y_test))

