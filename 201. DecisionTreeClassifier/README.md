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