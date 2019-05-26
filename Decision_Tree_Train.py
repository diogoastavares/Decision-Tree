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


def FindList (List, ObjectFind):
	myBool = False
	FlagTrue = 0 #Flag for when the cycle founds a match
	for x in range(0,len(List)):
		if List[x] == ObjectFind: #Object is in the list
			myBool = True
			FlagTrue = 1
		elif FlagTrue == 0:
			myBool = False
	return myBool


def ConvertFileMLAPLI(FileName, Scaling,FeatureList, ClassificationList, InputSize, Order, Mode):
	FileName = FileName + ".mlapi" #Add the dsl extension file
	file=open(FileName,'w') #File name
	dat = prunedTree.tree_ #Load tree object
	nodes = range(0, dat.node_count) #Number of nodes
	file.write("var SCALING_FACTOR = ") #Print file header
	if Scaling > 0: #Print Scaling factor
		file.write(str(Scaling))
		file.write("\n\n")
	else:
		print("Error Printing: Scaling must have a value higher than 0")
		file.close()
		return
	ListaNode = [] #List of possible nodes
	for i in nodes:
		if dat.children_left[i] >= 0: #Tree Node
			file.write("Node n" + str(i)) #Print node header
			file.write(" {\n") #Print function start
			file.write("\tnumber = " + str(i)) #Print node number
			file.write(";\n") #Print terminations
			file.write("\tfeature = \"" + FeatureList[dat.feature[i]]) #Print Node feature name
			file.write("\";\n") #Print terminations
			file.write("\tweight = " + str(int(dat.threshold[i]))) #Print node compare weight
			file.write(";\n") #Print terminations
			file.write("\ttrue node = " + str(dat.children_left[i])) #Print left child
			file.write(";\n") #Print terminations
			file.write("\tfalse node = " + str(dat.children_right[i])) #Print right child
			file.write(";\n") #Print terminations
			file.write("}\n") #Print funtion termination
			ListaNode.append(dat.children_left[i]) # List of possible left nodes branching
			ListaNode.append(dat.children_right[i]) # List of possible right nodes branching
		elif FindList(ListaNode, i): #Tree Leaf
			if ListaNode.index(i) == ValueError: #Not Found in the list
				ListaNode.append(i) #Add to the list the leaf
			ValueNumber = dat.value[i][0] #Conversion to array
			file.write("Node n" + str(i)) #Print node header
			file.write(" {\n") #Print funtion start
			file.write("\tnumber = " + str(i)) #Print node number
			file.write(";\n") #Print terminations
			if ValueNumber[0] > ValueNumber[1]: #Decision between the weight of the classifications
				file.write("\tclass = \"" + ClassificationList[1]) #Print Old
			else:
				file.write("\tclass = \"" + ClassificationList[0]) #Print Young
			file.write("\";\n") #Print terminations
			file.write("}\n") #Print funtion termination
	file.write("\n\nBinaryTree bt {\n\tinput_size = 1 x ")
	if int(InputSize) > 0: #Print Input
		file.write(str(InputSize))
	else:
		print("Error Printing: Input Size Must Be a value greater than 0")
		file.close()
		return
	file.write(";")
	file.write("\n\ttraversal type = #") #Print Type of traversal
	if Order == "Inorder":
		file.write("Inorder;")
	elif Order == "Postorder":
		file.write("Postorder;")
	else:
		print("Error Printing: Order must have the values of Inorder or Postorder")
		file.close()
		return
	file.write("\n\tmode = #") #Print accelarator mode
	if Mode == "On":
		file.write("AcceleratorOn;")
	elif Mode == "Off":
		file.write("AcceleratorOff;")
	else:
		print("Error Printing: Mode must have the values of On or Off")
		file.close()
		return
	file.write("\n\tfeatures = [")	
	for x in range(0,len(FeatureList)): #Print features
		file.write("\"")
		file.write(str(FeatureList[x]))
		if x != len(FeatureList) - 1:
			file.write("\",")
		else:
			file.write(";]")
	file.write("\n\tnodes list = [")
	ListaNode.sort()
	for x in range(0, len(ListaNode)): #Print nodes
		file.write("n")
		file.write(str(ListaNode[x]))
		if x != len(ListaNode) - 1:
			file.write(",")
		else:
			file.write("]")
	file.write("\n}")
	file.close()

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

#Convert to MLAPI file
FeatureList = ["Sex", "Lenght", "Diameter", "Height", "Whole Weight" , "Shucked Weight", "Viscera Weight", "Shell Weight"] #List of features of tree
ClassificationList = ["Young", "Old"] #List of possible classifications
Scaling = 10000;
InputSize = 8
Order = "Inorder"
Mode = "On"
FileName = "TreeFile"
ConvertFileMLAPLI(FileName, Scaling, FeatureList, ClassificationList, InputSize, Order, Mode) #Call for printing file
