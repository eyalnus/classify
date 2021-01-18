import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import datetime
from pandas.api.types import is_numeric_dtype
import sklearn as skl
from scipy.stats import skewnorm
import scipy.stats as stats
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


def bayes_plot(df,model="gnb",spread=30):
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
        clf = DecisionTreeClassifier()

    # Train Classifer
    clf = clf.fit(X_train, y_train)

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
    plt.show()

def tortuga():
    df = pd.read_csv('dataset-tortuga.csv')
    df_2 = df.copy()
    #print(df_2.iloc[:,0])
    # ind = df_2.sample(10).index
    # print(df_2.loc[ind,'Horizontal_Distance_To_Hydrology'])
    # df_2.dropna(axis=0, inplace=True)



    # sns.displot(data=df_2,x="var_2",hue='Segmentation')
    # plt.show()

    sns.scatterplot(data=df_2, x="NUM_COURSES_ADVANCED_DATASCIENCE", y="NUM_COURSES_ADVANCED_FRONTEND",hue="PROFILE")
    plt.show()
    #df_2.to_csv("weather.csv", index=False)
    #df_2['Date'] = (pd.to_datetime(df_2['Date']) - datetime.datetime(1970, 1, 1)).dt.total_seconds()
    #df_2 = pd.get_dummies(df_2, columns="WindGustDir,WindDir9am,WindDir3pm,RainToday".split(','))

    df_2.dropna(axis=0, inplace=True)
    print(df_2.columns)
    y = df_2["PROFILE"]  # Target variable
    X = df_2.drop(["PROFILE", "NAME","USER_ID","Unnamed: 0"], axis=1)


    #df_2 = pd.get_dummies(df_2, columns="PROFILE".split(','))
    #df_2.columns = df_2.columns.str[0] + df_2.columns.str[12]
    corr = df_2.corr()  # pd.concat([X,pd.get_dummies(df["source"])],axis=1).corr()
    sns.heatmap(corr)
    plt.show()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=1)  # 70% training and 30% test
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print("Tree")
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    print(metrics.classification_report(y_test, y_pred))
    plt.show()

    gnb = GaussianNB()
    gnb = gnb.fit(X_train, y_train)
    gnb_pred = gnb.predict(X_test)
    gnb_prob = gnb.predict_proba(X_test)
    print("GNB")
    print(metrics.classification_report(y_test, gnb_pred))

    print("Log loss:", metrics.log_loss(y_test, gnb_prob))

    gnb = GaussianNB()


    gnb = gnb.fit(X.loc[:, ["NUM_COURSES_ADVANCED_DATASCIENCE","NUM_COURSES_ADVANCED_FRONTEND"]], y)
    def numify(x):

        return np.where(gnb.classes_ == x)[0]

    Y = y.apply(numify)
    print(Y)
    x_min, x_max = X.loc[:, "NUM_COURSES_ADVANCED_DATASCIENCE"].min() - 1, X.loc[:, "NUM_COURSES_ADVANCED_DATASCIENCE"].max() + 1
    y_min, y_max = X.loc[:, "NUM_COURSES_ADVANCED_FRONTEND"].min() - 1, X.loc[:, "NUM_COURSES_ADVANCED_FRONTEND"].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2))

    print(np.c_[xx.ravel(), yy.ravel()].shape)
    print(yy.shape)
    Z = gnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    print(Z)
    ZZ= np.argmax(Z,axis=1)
    print(ZZ)
    # Put the result into a color plot
    ZZ = ZZ.reshape(xx.shape)
    plt.contourf(xx, yy, ZZ, cmap=plt.cm.twilight, alpha=0.3)
    plt.colorbar()
    hueorder = np.unique(df_2.PROFILE).tolist()
    hueorder.sort()
    print(hueorder)
    # Plot also the training points
    sns.scatterplot(data=df_2, x="NUM_COURSES_ADVANCED_DATASCIENCE", y="NUM_COURSES_ADVANCED_FRONTEND",hue="PROFILE",hue_order=hueorder,palette='twilight')

    #plt.scatter(X.loc[:, "NUM_COURSES_BEGINNER_DATASCIENCE"], X.loc[:, "AVG_SCORE_BACKEND"], c=Y[:])
    plt.xlabel('NUM_COURSES_ADVANCED_DATASCIENCE')
    plt.ylabel('NUM_COURSES_ADVANCED_FRONTEND')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()

    Y_pred = gnb.predict(X.loc[:, ["NUM_COURSES_ADVANCED_DATASCIENCE","NUM_COURSES_ADVANCED_FRONTEND"]])
    print("GNB")
    print(metrics.classification_report(y, Y_pred))



if __name__ == "__main__":

    df = pd.read_csv('dataset-tortuga.csv')
    df_2 = df.copy()
    df_2.fillna(0,inplace=True)


    df_front = df_2[(df_2["PROFILE"].str.find("front_end") != -1)].loc[:,["HOURS_FRONTEND","AVG_SCORE_FRONTEND","PROFILE"]]


    df_back = df_2[(df_2["PROFILE"].str.find("end") != -1)].loc[:,["HOURS_FRONTEND","AVG_SCORE_FRONTEND","PROFILE"]]

    df_all = df_2.loc[:,["HOURS_FRONTEND","AVG_SCORE_FRONTEND","PROFILE"]]

    #bayes_plot(df_front)
    #bayes_plot(df_back)
    #bayes_plot(df_all)

    #bayes_plot(df_front,"tree")
    #bayes_plot(df_back,"tree")
    #bayes_plot(df_all,"tree")

    df_f = df_2[(df_2["PROFILE"].str.find("front_end") != -1)].copy()
    #df_f['PROFILE'] = "frontend"
    df_b = df_2[(df_2["PROFILE"].str.find("backend") != -1)].copy()
    #df_b['PROFILE'] = "backend"
    df_diff = pd.concat([df_f,df_b],axis=0)
    df_diff = df_diff.loc[:,["AVG_SCORE_FRONTEND","AVG_SCORE_BACKEND","PROFILE"]]
    #bayes_plot(df_diff,spread=25)

    y = df_2["PROFILE"]  # Target variable
    X = df_2.drop(["PROFILE", "NAME","USER_ID","Unnamed: 0"], axis=1)

    corr = df_2.corr()  # pd.concat([X,pd.get_dummies(df["source"])],axis=1).corr()
    sns.heatmap(corr)
    plt.show()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=1)  # 70% training and 30% test
    gnb = GaussianNB()
    gnb = gnb.fit(X_train, y_train)
    gnb_pred = gnb.predict(X_test)
    gnb_prob = gnb.predict_proba(X_test)
    print("GNB")
    print(metrics.classification_report(y_test, gnb_pred))

    print("Log loss:", metrics.log_loss(y_test, gnb_prob))

    df_3 = df_2.copy()
    df_3.loc[(df_3["NUM_COURSES_BEGINNER_FRONTEND"] + df_3["NUM_COURSES_ADVANCED_FRONTEND"]) == 0,"H_FRONT_END"] = 0
    df_3.loc[(df_3["NUM_COURSES_BEGINNER_FRONTEND"] + df_3["NUM_COURSES_ADVANCED_FRONTEND"]) != 0,"H_FRONT_END"] = df_3["HOURS_FRONTEND"]/(df_3["NUM_COURSES_BEGINNER_FRONTEND"] + df_3["NUM_COURSES_ADVANCED_FRONTEND"])
    df_3.loc[(df_3["NUM_COURSES_BEGINNER_BACKEND"] + df_3["NUM_COURSES_ADVANCED_BACKEND"]) == 0, "H_BACKEND"] = 0
    df_3.loc[(df_3["NUM_COURSES_BEGINNER_BACKEND"] + df_3["NUM_COURSES_ADVANCED_BACKEND"]) != 0, "H_BACKEND"] = \
    df_3["HOURS_BACKEND"] / (df_3["NUM_COURSES_BEGINNER_BACKEND"] + df_3["NUM_COURSES_ADVANCED_BACKEND"])
    df_3.loc[(df_3["NUM_COURSES_BEGINNER_DATASCIENCE"] + df_3["NUM_COURSES_ADVANCED_DATASCIENCE"]) == 0, "H_DATASCIENCE"] = 0
    df_3.loc[(df_3["NUM_COURSES_BEGINNER_DATASCIENCE"] + df_3["NUM_COURSES_ADVANCED_DATASCIENCE"]) != 0, "H_DATASCIENCE"] = \
    df_3["HOURS_DATASCIENCE"] / (df_3["NUM_COURSES_BEGINNER_DATASCIENCE"] + df_3["NUM_COURSES_ADVANCED_DATASCIENCE"])


    df_3.drop(['NUM_COURSES_BEGINNER_FRONTEND','NUM_COURSES_ADVANCED_FRONTEND',
               'HOURS_FRONTEND','NUM_COURSES_BEGINNER_BACKEND','NUM_COURSES_ADVANCED_BACKEND','HOURS_BACKEND',
               'NUM_COURSES_BEGINNER_DATASCIENCE','NUM_COURSES_ADVANCED_DATASCIENCE','HOURS_DATASCIENCE'],axis=1, inplace=True)

    y = df_3["PROFILE"]  # Target variable
    X = df_3.drop(["PROFILE", "NAME","USER_ID","Unnamed: 0"], axis=1)

    corr = df_3.corr()  # pd.concat([X,pd.get_dummies(df["source"])],axis=1).corr()
    sns.heatmap(corr)
    plt.show()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=1)  # 70% training and 30% test
    gnb = GaussianNB()
    gnb = gnb.fit(X_train, y_train)
    gnb_pred = gnb.predict(X_test)
    gnb_prob = gnb.predict_proba(X_test)
    print("GNB")
    print(metrics.classification_report(y_test, gnb_pred))

    print("Log loss:", metrics.log_loss(y_test, gnb_prob))
    pass