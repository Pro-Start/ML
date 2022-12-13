### Importing Libraries

```python
import pandas as pd
df = pd.read_csv('vgsales.csv')
```

### Assigning **X and y** variables

```python
X = df.drop(columns=['genre'])
y = df['genre']
```

### Decison Tree Classifier

```python
from sklearn.tree import DecisionTreeClassifier

model1 = DecisionTreeClassifier()
model1.fit(X.values, y)

prediction1 = model1.predict([[21, 1], [22, 0]])
prediction1
```

### Training the model

```python
from sklearn.model_selection import train_test_split
```

### Splitting the dataset into the Training set and Test set

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### Training the model

```python
model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)
prediction2 = model2.predict(X_test)
prediction2
```