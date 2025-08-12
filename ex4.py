import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree

data=pd.read_csv('ex4.csv')
df=pd.DataFrame(data)

x=df[['Study_Hours','Attendance']]
y=df['Result']
dtc=DecisionTreeClassifier(criterion='entropy',random_state=0)
dtc.fit(x,y)
plt.figure(figsize=(8,6))
plot_tree(dtc,feature_names=['Study_Hours','Attendance'],class_names=['Fail','Pass'],filled=True)
plt.show()

new=[[5,85]]
pred=dtc.predict(new)
print("Prediction for new student:","1" if pred[0]==1 else "0")
