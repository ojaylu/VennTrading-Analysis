from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import *
from sklearn import svm
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from calc import *
from sklearn import tree
import dtreeviz
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def decisiontree(data):

    #clf = DecisionTreeClassifier(criterion='gini', splitter='best',max_depth=100,min_samples_split=2,min_samples_leaf=6,min_weight_fraction_leaf=0.0,max_features=None,random_state=42,max_leaf_nodes=None,min_impurity_decrease=0.0)
    clf = DecisionTreeClassifier()
    #clf = KNeighborsClassifier(n_neighbors=5)
    #clf = svm.SVC(probability = True)

    data = calc(data)
    #data['RSI_31'].fillna(value=data['RSI_31'].mean(), inplace=True)
    predictors_list = ['close', 'RSI_14', 'EMA10', 'EMA30', 'macd','OBV', 'ATR','ClgtEMA10', 'EMA10gtEMA30', 'MACDSIGgtMACD', '%k', '%d' ]
    X = data[predictors_list]
    y = data[['prediction']]

    X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X, y, test_size=0.3, random_state=432, stratify=y, shuffle=True)
    clf = clf.fit(X_cls_train.values, y_cls_train.values)
    explainer = LimeTabularExplainer(np.array(X_cls_train), feature_names=X_cls_train.columns,class_names=['prediction'])
    exp = explainer.explain_instance(X_cls_test.iloc[4], clf.predict_proba)
    exp.show_in_notebook(show_table=True, show_all=False)
    #fig = exp.as_pyplot_figure()
    #fig.savefig('lime_report.jpg')
    #viz = dtreeviz(clf, X, y,target_name="prediction",feature_names=predictors_list,class_names=[-1,0,1])
    # dot_data = tree.export_graphviz(clf, out_file=None,filled=True,feature_names=predictors_list)
    # print(graphviz.Source(dot_data).view())
    y_cls_pred = clf.predict(pd.DataFrame(np.array(X_cls_test)))

    # from matplotlib.colors import ListedColormap
    # X_set, y_set = X_cls_test, y_cls_test
    # X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop= X_set[:,0].max()+1, step = 0.01),np.arange(start = X_set[:,1].min()-1, stop= X_set[:,1].max()+1, step = 0.01))
    # plt.contourf(X1,X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75, cmap = ListedColormap(("red","green")))
    # plt.xlim(X1.min(), X1.max())
    # plt.ylim(X2.min(), X2.max())
    # for i,j in enumerate(np.unique(y_set)):
    #     plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1], c = ListedColormap(("red","green"))(i),label = j)
    # plt.title("Decision Tree(Test set)")
    # plt.xlabel("Age")
    # plt.ylabel("Estimated Salary")
    # plt.legend()
    # plt.show()

    report = classification_report(y_cls_test, y_cls_pred)
    accuracy = accuracy_score(y_cls_test, y_cls_pred)*100

    importances = clf.feature_importances_
    indices = np.argsort(importances)
    features = data.columns
    plt.cla()
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='g', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

    cm = confusion_matrix(y_cls_test, y_cls_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
    plt.cla()
    disp.plot()
    plt.show()
    class_names = [1, 0,-1]
    # disp = ConfusionMatrixDisplay(clf, X_cls_test, y_cls_test, display_labels=class_names)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X.values, plot_type="bar", class_names= class_names, feature_names = X.columns)
    shap.summary_plot(shap_values[1], X.values, feature_names = X.columns)
    shap.dependence_plot(0, shap_values[0], X.values, feature_names=X.columns)

    i=8
    shap.force_plot(explainer.expected_value[0], shap_values[0][i], X.values[i], feature_names = X.columns)

    row = 8
    shap.waterfall_plot(shap.Explanation(values=shap_values[0][row], base_values=explainer.expected_value[0], data=X_cls_test.iloc[row],  
                                         feature_names=X_cls_test.columns.tolist()))

    print('the prediction by classifier is',y_cls_pred)
    print('the true future signal should be', [x for x in y_cls_test['prediction']])
    print(report)
    print('accuracy score is', accuracy)

    return data, clf