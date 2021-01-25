#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install pandas')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install scipy')
get_ipython().system('pip install pydotplus')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import sklearn as skl
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import pydotplus


# # Classification using sklearn
# Scikit-learn (sklearn) is a Python library dedicated to machine learning. It contains classifier and regression algorithm objects which implement an API for training and predicting models. Additionally, it contains some methods for data manipulation and performance metric measuring of predictrive models.

# ### Quick example - Iris dataset

# In[ ]:


iris = sns.load_dataset('iris') #seaborn has some built in datasets
iris.head()


# In[ ]:


sns.pairplot(iris, hue='species', height=1.5);


# ##### Splitting the data set into feature vector X and target variable y

# In[ ]:


X_iris = iris.drop('species', axis=1)
print(X_iris.shape)
y_iris = iris['species']
print(y_iris.shape)


# ##### Splitting the data set into training and test sets. By default, test set size is 25% of data set.

# In[ ]:


#from sklearn.model_selection import train_test_split 
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,
                                                random_state=1)


# ##### Training and predicting using Naive Bayes classifier

# In[ ]:


#from sklearn.naive_bayes import GaussianNB # 1. choose model class
model = GaussianNB()                       # 2. instantiate model
model.fit(Xtrain, ytrain)                  # 3. fit model to data
y_model = model.predict(Xtest)             # 4. predict on new data (output is numpy array)

ypred = pd.Series(y_model,name="prediction")
predicted = pd.concat([Xtest.reset_index(),ytest.reset_index(),ypred],axis=1)
predicted


# ##### Calculate the accuracy as an average of accuracy per class

# In[ ]:


#from sklearn import metrics
metrics.accuracy_score(ytest, y_model)


# ###### What happens if we select less columns?

# In[ ]:


X_iris = iris.drop(['species','petal_length','petal_width'], axis=1)
y_iris = iris['species']
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,
                                                random_state=1)
model = GaussianNB()                       # 2. instantiate model
model.fit(Xtrain, ytrain)                  # 3. fit model to data
y_model = model.predict(Xtest)             # 4. predict on new data (output is numpy array)

ypred = pd.Series(y_model,name="prediction")
predicted = pd.concat([Xtest.reset_index(),ytest.reset_index(),ypred],axis=1)
print(metrics.accuracy_score(ytest, y_model))

predicted


# In[ ]:


X_iris = iris.drop(['species','sepal_length','sepal_width'], axis=1)
y_iris = iris['species']
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,
                                                random_state=1)
model = GaussianNB()                       # 2. instantiate model
model.fit(Xtrain, ytrain)                  # 3. fit model to data
y_model = model.predict(Xtest)             # 4. predict on new data (output is numpy array)

ypred = pd.Series(y_model,name="prediction")
predicted = pd.concat([Xtest.reset_index(),ytest.reset_index(),ypred],axis=1)
print(metrics.accuracy_score(ytest, y_model))
predicted


# In[ ]:


def bayes_plot(df,model="gnb",spread=30):
    df.dropna()
    colors = 'seismic'
    col1 = df.columns[0]
    col2 = df.columns[1]
    target = df.columns[2]
    sns.scatterplot(data=df, x=col1, y=col2,hue=target)
    plt.show()
    y = df[target]  # Target variable
    X = df.drop(target, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)  # 70% training and 30% test

    clf = GaussianNB()
    if (model != "gnb"):
        clf = DecisionTreeClassifier(max_depth=model)
    clf = clf.fit(X_train, y_train)
    
    # Train Classifer
    

    prob = len(clf.classes_) == 2

    # Predict the response for test dataset

    y_pred = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))

    hueorder = clf.classes_
    def numify(val):
        return np.where(clf.classes_ == val)[0]

    Y = y.apply(numify)
    x_min, x_max = X.loc[:, col1].min() - 1, X.loc[:, col1].max() + 1
    y_min, y_max = X.loc[:, col2].min() - 1, X.loc[:, col2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2))

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    if prob:

        Z = Z[:,1]-Z[:,0]
    else:
        colors = "Set1"
        Z = np.argmax(Z, axis=1)


    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=colors, alpha=0.5)
    plt.colorbar()
    if not prob:
        plt.clim(0,len(clf.classes_)+3)
    sns.scatterplot(data=df[::spread], x=col1, y=col2, hue=target, hue_order=hueorder,palette=colors)
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    plt.show()


# In[ ]:


bayes_plot(pd.concat([X_iris,y_iris],axis=1),spread=1)


# In[ ]:


iris_poor = iris.drop(['petal_length','petal_width'], axis=1)
bayes_plot(iris_poor,spread=1)


# In[ ]:


bayes_plot(pd.concat([X_iris,y_iris],axis=1),model=4,spread=1)


# In[ ]:


bayes_plot(pd.concat([X_iris,y_iris],axis=1),model=2,spread=1)


# In[ ]:


from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=1.0)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow');


# In[ ]:


blob_df = pd.DataFrame(X,columns=['X','Y'])
blob_series = pd.Series(y,name="target")
print(pd.concat([blob_df,blob_series],axis=1))
bayes_plot(pd.concat([blob_df,blob_series],axis=1),model=3,spread=1)


# In[ ]:


bayes_plot(pd.concat([blob_df,blob_series],axis=1),model=5,spread=1)


# *Note: we have a slight drop in accuracy, perhaps due to overfitting*

# In[ ]:


from io import StringIO
get_ipython().system('conda install -y python-graphviz')
from IPython.display import Image
from sklearn.tree import export_graphviz


# In[ ]:


from sklearn.inspection import permutation_importance


tech = pd.read_csv('dataset-tortuga.csv')
tech.dropna(inplace=True)
X = tech.drop(["PROFILE", "NAME","USER_ID","Unnamed: 0"],axis=1)
y = tech['PROFILE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)  # 70% training and 30% test
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

result = permutation_importance(clf, X, y, n_repeats=10,random_state=0)
importance = zip(X.columns,result['importances_mean'])
# summarize feature importance
for i,v in importance:
    print('Feature: %s, Score: %.5f' % (i,v))
# plot feature importance
print(len(X.columns),[x[1] for x in importance])
plt.bar(range(len(X.columns)), result['importances_mean'])
plt.xticks(ticks=range(len(X.columns)),labels=X.columns, rotation=90)
plt.show()

y_pred = clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))

dot_data=StringIO()
export_graphviz(clf,out_file=dot_data,filled=True,rounded=True,feature_names=X.columns,class_names=clf.classes_)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Tech.png')
Image(graph.create_png())


# In[ ]:


#highest importance

X_2d = X[["NUM_COURSES_BEGINNER_BACKEND","AVG_SCORE_DATASCIENCE"]]
y_2d = tech['PROFILE']
X_train, X_test, y_train, y_test = train_test_split(X_2d, y_2d, test_size=0.3,random_state=1)  # 70% training and 30% test
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))

dot_data=StringIO()
export_graphviz(clf,out_file=dot_data,filled=True,rounded=True,feature_names=X_2d.columns,class_names=clf.classes_)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Tech2d.png')
Image(graph.create_png())


# *Note: this does not mean that the top "important" features will always yield the best results. Remember that some split decisions are random and will affect importance.*

# In[ ]:



bayes_plot(pd.concat([X_2d,y_2d],axis=1),model=20,spread=10)


# In[ ]:


#lowest importance 
X_2d = X[["NUM_COURSES_ADVANCED_DATASCIENCE","NUM_COURSES_ADVANCED_FRONTEND"]]
y_2d = tech['PROFILE']
X_train, X_test, y_train, y_test = train_test_split(X_2d, y_2d, test_size=0.3,random_state=1)  # 70% training and 30% test
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)  # 70% training and 30% test
clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))

dot_data=StringIO()
export_graphviz(clf,out_file=dot_data,filled=True,rounded=True,feature_names=X.columns,class_names=clf.classes_)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Tech_depth4.png')
Image(graph.create_png())


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)  # 70% training and 30% test
clf = DecisionTreeClassifier(max_depth=6)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))

dot_data=StringIO()
export_graphviz(clf,out_file=dot_data,filled=True,rounded=True,feature_names=X.columns,class_names=clf.classes_)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Tech_depth6.png')
Image(graph.create_png())


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)  # 70% training and 30% test
clf = DecisionTreeClassifier(min_impurity_decrease=0.003)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))

dot_data=StringIO()
export_graphviz(clf,out_file=dot_data,filled=True,rounded=True,feature_names=X.columns,class_names=clf.classes_)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Tech_impurity.png')
Image(graph.create_png())


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)  # 70% training and 30% test
clf = DecisionTreeClassifier(max_leaf_nodes =50)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))

dot_data=StringIO()
export_graphviz(clf,out_file=dot_data,filled=True,rounded=True,feature_names=X.columns,class_names=clf.classes_)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Tech_maxleaf.png')
Image(graph.create_png())


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)  # 70% training and 30% test
clf = DecisionTreeClassifier(min_samples_split=0.01)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))

dot_data=StringIO()
export_graphviz(clf,out_file=dot_data,filled=True,rounded=True,feature_names=X.columns,class_names=clf.classes_)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Tech_minsamplesplit.png')
Image(graph.create_png())


# In[ ]:


X_cut = X.drop(["NUM_COURSES_ADVANCED_DATASCIENCE","NUM_COURSES_ADVANCED_FRONTEND"],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_cut, y, test_size=0.3,random_state=1)  # 70% training and 30% test
clf = DecisionTreeClassifier(min_samples_split=0.01)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))

dot_data=StringIO()
export_graphviz(clf,out_file=dot_data,filled=True,rounded=True,feature_names=X_cut.columns,class_names=clf.classes_)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Tech_minsamplesplit2.png')
Image(graph.create_png())


# #### Filling NaNs

# In[ ]:


tech = pd.read_csv('dataset-tortuga.csv')

def nullify(x):
    if np.random.random() < 0.5:
        return np.NaN
    else:
        return x

tech['HOURS_BACKEND'] = tech['HOURS_BACKEND'].apply(nullify)
tech['HOURS_FRONTEND'] = tech['HOURS_FRONTEND'].apply(nullify)
tech['HOURS_DATASCIENCE'] = tech['HOURS_DATASCIENCE'].apply(nullify)



techna = tech.dropna()
X = techna.drop(["PROFILE", "NAME","USER_ID","Unnamed: 0"],axis=1)
y = techna['PROFILE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)  # 70% training and 30% test
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))


# In[ ]:



techna = tech.copy()
techna['HOURS_BACKEND'].fillna(tech['HOURS_BACKEND'].mean(),inplace=True)
techna['HOURS_FRONTEND'].fillna(tech['HOURS_FRONTEND'].mean(),inplace=True)
techna['HOURS_DATASCIENCE'].fillna(tech['HOURS_DATASCIENCE'].mean(),inplace=True)



techna = techna.dropna()
X = techna.drop(["PROFILE", "NAME","USER_ID","Unnamed: 0"],axis=1)
y = techna['PROFILE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)  # 70% training and 30% test
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))


# ## Classifying the Titanic Survivors

# In[ ]:


titanic = pd.read_csv('titanic.csv')
titanic
titanic.isna().sum()


# In[ ]:


titanic_data = titanic.drop("PassengerId,Name,Ticket,Fare,Cabin".split(','),axis=1)
titanic_data["Age"].fillna(titanic_data["Age"].mean(),inplace=True)
titanic_data.dropna(inplace=True)
X = titanic_data.drop(["Survived"],axis=1)
X = pd.get_dummies(X)

y = titanic_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)  # 70% training and 30% test
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))

result = permutation_importance(clf, X, y, n_repeats=1,random_state=0)
importance = zip(X.columns,result['importances_mean'])
# summarize feature importance
for i,v in importance:
    print('Feature: %s, Score: %.5f' % (i,v))
# plot feature importance
print(len(X.columns),[x[1] for x in importance])
plt.bar(range(len(X.columns)), result['importances_mean'])
plt.xticks(ticks=range(len(X.columns)),labels=X.columns, rotation=90)
plt.show()


# In[ ]:


X_2d = X[["Age","Sex_female"]]
y_2d = y
bayes_plot(pd.concat([X_2d,y_2d],axis=1),model=20,spread=10)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.3,random_state=1)  # 70% training and 30% test
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
dot_data=StringIO()
export_graphviz(clf,out_file=dot_data,filled=True,rounded=True,feature_names=X_2d.columns,class_names=["0","1"])
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('titanic.png')
Image(graph.create_png())


# In[ ]:




