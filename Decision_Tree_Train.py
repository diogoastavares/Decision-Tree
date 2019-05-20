import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mp
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler

def convertSex():
    df['Sex'] = df['Sex'].map({'M': 0, 'F': 1, 'I': 2})

def multiply_by_Scalar(conv_array, scalar):
    for i in conv_array:
        df[i] = df[i].apply(lambda x: x*scalar*255/65535)

def treat_data():
    global df
    df.round()
    df=df.apply(np.int8)

def prune(tree):
    dat = tree.tree_
    nodes = range(0, dat.node_count)
    for i in nodes:
	    dat.threshold[i]=int(dat.threshold[i])
    ls = dat.children_left
    rs = dat.children_right
    classes = [[list(e).index(max(e)) for e in v] for v in dat.value]

    leaves = [(ls[i] == rs[i]) for i in nodes]

    LEAF = -1
    for i in reversed(nodes):
        if leaves[i]:
            continue
        if leaves[ls[i]] and leaves[rs[i]] and classes[ls[i]] == classes[rs[i]]:
            ls[i] = rs[i] = LEAF
            leaves[i] = True
    return tree

f = 'abalone.data'
sep = ','
df = read_csv(f, sep, header = None, names = ['Sex','Length','Diameter','Height','Whole Weight', 'Shucked Weight','Viscera Weight','Shell Weight','Rings'])

#Eliminate zeros and outliers
df = df[df.Height > 0]
df = df[df.Height < 0.4]
df = df[df.Rings >= 3]
df = df[df.Rings <= 22]


#Standardization
'''
for label in ['Length','Diameter','Height','Whole Weight', 'Shucked Weight','Viscera Weight','Shell Weight']:
    df[label] = df[label].apply(lambda x: abs(x-df[label].mean())/df[label].std())
'''

#Normalization 

for label in ['Length','Diameter','Height','Whole Weight', 'Shucked Weight','Viscera Weight','Shell Weight']:
    df[label] = df[label].apply(lambda x: abs(x-(df[label].mean()/(df[label].max()-df[label].min()))))


#Convert Data
convertSex()
multiply_by_Scalar(['Length','Diameter','Height','Whole Weight', 'Shucked Weight','Viscera Weight','Shell Weight'], 10000)
treat_data()
plt.show()
#Categorization

df['Age']= df.Rings
df['Age'] = df['Age'].map(lambda x: 'Young' if x<=9 else 'Old')

df=df.drop(columns='Rings')
df = df[['Age','Sex','Length','Diameter','Height','Whole Weight', 'Shucked Weight','Viscera Weight','Shell Weight']]


#Induction
X = df.values[:, 1:9]
Y = df.values[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=11)

model = DecisionTreeClassifier(criterion = 'gini', splitter='best', max_depth=20, min_samples_leaf=50, random_state=11)

#Training
model.fit(X_train, y_train)

#Cross validation
cross_score = cross_val_score(model, X_train, y_train, cv=5)
print('cross score: ', cross_score)

#Tests without pruning
y_pred = model.predict(X_test)
print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)
CM=confusion_matrix(y_test,y_pred)
print("Accuracy with Confusion Matrix ", (CM.trace())/(CM.sum())*100)

dot_data = tree.export_graphviz(model, out_file='tree_8bit.dot', feature_names=['Sex','Length','Diameter','Height','Whole Weight', 'Shucked Weight','Viscera Weight','Shell Weight'], label= 'None', class_names=df.Age, filled=False, rounded=False, special_characters=False)

#Pruning
prunedTree= prune(model)

#Tests with pruning
pruned_y_pred = prunedTree.predict(X_test)

print ("Pruned Accuracy is ", accuracy_score(y_test,pruned_y_pred)*100)
pruned_CM=confusion_matrix(y_test,pruned_y_pred)
print("Pruned Accuracy with Confusion Matrix ", (pruned_CM.trace())/(pruned_CM.sum())*100)

pruned_dot_data = tree.export_graphviz(prunedTree, out_file='pruned_tree_8bit.dot', feature_names=['Sex','Length','Diameter','Height','Whole Weight', 'Shucked Weight','Viscera Weight','Shell Weight'], label= 'None', class_names=df.Age, filled=False, rounded=False, special_characters=False)

X_test_frame=pd.DataFrame(data=X_test, columns=['Sex','Length','Diameter','Height','Whole Weight', 'Shucked Weight','Viscera Weight','Shell Weight'])
Y_test_frame=pd.DataFrame(data=np.array(y_test), columns=['Age'])
test_data=pd.concat([Y_test_frame,X_test_frame], axis=1)
print (test_data)
test_data.to_csv('test_data_decision_tree_8bit.csv', sep = ',')