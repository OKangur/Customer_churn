import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import warnings

plt.style.use('fivethirtyeight')
warnings.filterwarnings("ignore")

# I'm using 10 random train test splits to measure how good the program is to predict customer churn.
number_of_tests = 10

for test in range(number_of_tests):
    # Read in data
    data = pd.read_csv('Data.csv')
    if test == 0:
        print(data.head())
        print(data.info())

    # Remove Phone and State columns
    data = data.drop(['Phone'], axis=1)
    data = data.drop(['State'], axis=1)

    # Create new columns - Domestic_Mins, Total_Mins, Domestic_Calls, Total_Calls, Domestic_Charge, Total_Charge
    data['Domestic_Mins'] = data['Day Mins'] + data['Eve Mins'] + data['Night Mins']
    data['Total_Mins'] = data['Day Mins'] + data['Eve Mins'] + data['Night Mins'] + data['Intl Mins']
    data['Domestic_Calls'] = data['Day Calls'] + data['Eve Calls'] + data['Night Calls']
    data['Total_Calls'] = data['Day Calls'] + data['Eve Calls'] + data['Night Calls'] + data['Intl Calls']
    data['Domestic_Charge'] = data['Day Charge'] + data['Eve Charge'] + data['Night Charge']
    data['Total_Charge'] = data['Day Charge'] + data['Eve Charge'] + data['Night Charge'] + data['Intl Charge']
    data['Total_Charge_Times_Custserv_Calls'] = (data['Day Charge'] + data['Eve Charge'] + data['Night Charge'] + data['Intl Charge']) * data['CustServ Calls']

    # Makes train test split (2/3 and 1/3) with equal percentage of customer churn

    data_churn_0 = data[data['Churn']==0]
    data_churn_1 = data[data['Churn']==1]

    data_churn_0_train = data_churn_0.sample(frac=2/3.0)
    data_churn_1_train = data_churn_1.sample(frac=2/3.0)

    data_churn_0_test = data_churn_0.drop(data_churn_0_train.index)
    data_churn_1_test = data_churn_1.drop(data_churn_1_train.index)

    X_train = pd.concat([data_churn_0_train,data_churn_1_train],axis=0)
    y_train = X_train['Churn']


    X_test = pd.concat([data_churn_0_test,data_churn_1_test],axis=0)
    y_test = X_test['Churn']
    X_test = X_test.drop(['Churn'], axis=1)

    data = data.drop(['Churn'], axis=1)

    # Chart column vs. Churn
    columns = list(X_train.columns.values)
    if test == 0:
        for column in columns:
            fig = plt.figure(figsize=(15,8))
            churn_acc_len_1 = X_train[X_train['Churn'] == 1][column]
            churn_acc_len_0 = X_train[X_train['Churn'] == 0][column]
            plt.hist([churn_acc_len_1, churn_acc_len_0],
                           stacked=True, color=['g','r'],
                           bins = 10, label=['1','0'])
            plt.xlabel(column)
            plt.ylabel('Churn')
            plt.title('Churn vs.' + column)
            plt.legend()
            #plt.show()
    X_train = X_train.drop(['Churn'], axis=1)
    columns.remove('Churn')

    # List of all classifier candidates saved in a list with parameters for grid
    # search [name, classifier object, parameters for grid search] and name.
    # Classifieers are RandomForestClassifier, ExtraTreesClassifier, DecisionTreeClassifier,
    # KNeighborsClassifier

    candidates = []

    param_grid_forest = {
                 'min_samples_split': [10,20],
                 'max_depth': [7,9],
                 'criterion':['gini','entropy'],
                 'max_leaf_nodes': [30]
                 }

    candidates.append(["RandomForest",
                       RandomForestClassifier(n_jobs=-1,
                                              oob_score=True,
                                              min_samples_leaf=10,bootstrap=True),
                       param_grid_forest])

    candidates.append(["ExtraTrees",
                       ExtraTreesClassifier(n_jobs=-1,
                                              oob_score=True,
                                              min_samples_leaf=10,bootstrap=True),
                       param_grid_forest])

    param_grid_decisiontree = {
                'criterion':['gini','entropy'],
                'max_depth': [3,4,5],
                'min_samples_split': [10,20,30],
                'max_leaf_nodes': [5,10,20]
                }

    candidates.append(["ExtraTrees",
                       DecisionTreeClassifier(min_samples_leaf=5),
                       param_grid_decisiontree])

    param_grid_knn = {"n_neighbors": [5, 7, 9, 11, 13],
                        'n_jobs': [-1],
                      'weights': ['uniform', 'distance'],
                        'metric': ['euclidean', 'minkowski']}

    candidates.append(["kNN",
                       KNeighborsClassifier(),
                       param_grid_knn])

    # Starting parameters

    best_quality = 0.0
    best_classifier = None
    best_param = []

    # Performs 5 fold grid search

    for item in candidates:
        model_parameters = GridSearchCV(estimator=item[1], param_grid=item[2], cv= 5)
        model_parameters.fit(X_train,y_train)

        if round(model_parameters.best_score_,4)*100.0 > best_quality:
            best_quality = round(model_parameters.best_score_,4)*100.0
            best_classifier = item[1]
            best_param = model_parameters.best_params_

    print('Best prediction accuracy: ' + str(round(best_quality, 4)) + '%')
    print('Best_classifier: ' + str(best_classifier))
    print('Best_parameters: ' + str(best_param))
    print(' ')

    # Train the model on training set with best parameters
    model = best_classifier
    model.fit(X_train, y_train)

    # Calculate feature importance
    feature_importances = list(model.feature_importances_)
    if test == 0:
        feature_importances_dict = {}
        roc_curve = 0
        pred = 0 #sum of best predictions
        all_pred = []
        for i in range(len(columns)):
            feature_importances_dict[columns[i]] = round(feature_importances[i]*100,2)
    else:
        for i in range(len(columns)):
            feature_importances_dict[columns[i]] += round(feature_importances[i]*100,2)

    # Sort and print feature importance
    feature_importances_sorted = sorted(feature_importances_dict.items(), key=lambda x:x[1], reverse=True)

    #for item in feature_importances_sorted:
        #print(item)

    # File to vizualise decicion tree
    # export_graphviz(model,
    #                 class_names='Churn',
    #                 feature_names=columns,
    #                 filled=True,
    #                 rounded=True,
    #                 )

    # Calculate and print ROC curve
    prediction = model.predict(X_train)
    print('Area under the roc curve', round(roc_auc_score(y_train,prediction),4))
    roc_curve += round(roc_auc_score(y_train,prediction),4)

    # Result
    right_pred = 0
    test_preds = model.predict(X_test)
    for i in range(len(test_preds)):
        if test_preds[i] == y_test.values[i]:
            right_pred += 1
    print('Right guessed: '  + str(round(float(right_pred/len(test_preds))*100,2)) + '%')

    # Save test result
    pred += round(float(right_pred/len(test_preds))*100,2)
    all_pred.append(round(float(right_pred/len(test_preds))*100,2))

# Print average prediction and error
print('')
print('Average performance of tests: ' + str(round(pred / number_of_tests, 4)) + '% ROC: ' + str(round(roc_curve / number_of_tests, 4)))

average_error = 0
for i in range(number_of_tests):
    average_error += abs(all_pred[i] - pred / number_of_tests)
print('Average error: ' + str(round(average_error / number_of_tests, 4)) + '%')
