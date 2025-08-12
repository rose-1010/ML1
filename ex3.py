import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
df=pd.read_csv('ex3.csv')
df['gender'] = LabelEncoder().fit_transform(df['gender'])
x=df[['age', 'gender','bmi', 'blood_pressure', 'cholestrol', 'condition']]
y=df['condition']
scalar = StandardScaler()
xscale = scalar.fit_transform(x)
xtr, xte,ytr, yte = train_test_split(xscale, y, test_size=0.2, random_state=42)
model=LogisticRegression()
model.fit(xtr, ytr)
ypr=model.predict(xte)
yprob=model.predict_proba(xte)[:, 1]
print("Accuracy:",accuracy_score(yte, ypr))
print("classification report:\n", classification_report(yte, ypr, zero_division=1))
print("Confusion:\n", confusion_matrix(yte, ypr))
cm= confusion_matrix(yte,ypr)
disp= ConfusionMatrixDisplay(confusion_matrix=cm)
import matplotlib.pyplot as plt
disp.plot()
disp.plot(cmap='Blues')
plt.title("Confusion matrix")
plt.show()
new=pad.DataFrame([[60, 1, 27, 130, 200]],columns=['age', 'gender', 'bmi', 'blood_pressure', 'cholestrol'])
newscale=scaler.transform(new)
newcondition=model.predict_proba(newscale)[0][1]
print(f"probability of developing the condition : {newcondition:.2f}")
