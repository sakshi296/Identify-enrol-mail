#!/usr/bin/python

import sys
import pickle


import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.grid_search import GridSearchCV
from numpy import mean
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import naive_bayes
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

###################### Task 1: Select what features you'll use. ############################
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['salary', 'deferral_payments', 'total_payments', \
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', \
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', \
'long_term_incentive', 'restricted_stock', 'director_fees'] 
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', \
'from_this_person_to_poi', 'shared_receipt_with_poi']
target_label = ['poi'] # You will need to use more features

features_list=target_label + email_features + financial_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#print data_dict

# total no. of datapoints
print "Total no. of data points:",len(data_dict)

# allocation across classes
poi = 0
for person in data_dict:
    if data_dict[person]["poi"]==True:
        poi+=1

print "total no. of poi's:",poi
print "total no. of non-poi's",len(data_dict)-poi

# no. of features used
all_features=data_dict[data_dict.keys()[0]].keys()
print "The no. of features for each person in the dataset, and features that are used:",len(all_features),len(features_list)
# Calculate or find if there are any missing values
missing_values={}
for feature in all_features:
    missing_values[feature]=0
for person in data_dict:
    for feature in data_dict[person]:
        if data_dict[person][feature]=="NaN":
            missing_values[feature]+=1
print "The no. of missing values for each feature:"
for feature in missing_values:
    print feature,":\t",missing_values[feature]



############################# Task 2: Remove outliers ############

#for key in ["TOTAL", "THE TRAVEL AGENCY IN THE PARK", "LOCKHART EUGENE E"]:

def PlotOutlier(data_dict,feature_x,feature_y):
    data = featureFormat(data_dict,[feature_x,feature_y,'poi'])
    for point in data:
        x=point[0]
        y=point[1]
        poi=point[2]
        if poi:
            color='red'
        else:
            color='blue'
        plt.scatter(x,y,color=color)
    plt.xlabel("feature_x")
    plt.ylabel("feature_y")
    plt.show()

# visualize outliers
print PlotOutlier(data_dict,"total_payments","total_stock_value")
print PlotOutlier(data_dict,"from_poi_to_this_person","from_this_person_to_poi")
print PlotOutlier(data_dict,"salary","bonus")
print PlotOutlier(data_dict,"total_payments","other")    
    
data_dict.pop('TOTAL', 0)

# function to remove outliers
def removeoutlier(dict_object,keys):
    for key in keys:
        dict_object.pop(key,0)

outlier=['TOTAL','THE TRAVEL AGENCY IN THE PARK','LOCKHART EUGENE E']
removeoutlier(data_dict,outlier)
        

################# Task 3: Create new feature(s)  #############################3
### Store to my_dataset for easy export below.


my_dataset = data_dict
for person in my_dataset:
    msg_from_poi=my_dataset[person]['from_poi_to_this_person']
    to_msg=my_dataset[person]['to_messages']
    if msg_from_poi !='NaN' and to_msg != 'NaN':
        my_dataset[person]['msg_from_poi_ratio']=msg_from_poi/float(to_msg)
    else:
        my_dataset[person]['msg_from_poi_ratio']=0
    msg_to_poi=my_dataset[person]['from_this_person_to_poi']
    from_msg=my_dataset[person]['from_messages']
    if msg_to_poi != 'NaN' and from_msg != 'NaN':
        my_dataset[person]['msg_to_poi_ratio']=msg_to_poi/float(from_msg)
    else:
        my_dataset[person]['msg_to_poi_ratio']=0

new_features_list=features_list+ ['msg_to_poi_ratio','msg_from_poi_ratio']
 


        
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Select the best features: 
#Removes all features whose variance is below 80% 
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
features = sel.fit_transform(features)

#Removes all but the k highest scoring features

#The optimal k value will be in this dictionary.


from sklearn.feature_selection import f_classif

k = 7
selector = SelectKBest(f_classif, k=7)
selector.fit_transform(features, labels)
print("Best features:")
scores = zip(new_features_list[1:],selector.scores_)
sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
print sorted_scores
optimized_features_list = target_label + list(map(lambda x: x[0], sorted_scores))[0:k]
print(optimized_features_list)

# Extract from dataset without new features
data = featureFormat(my_dataset, optimized_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)
# Extract from dataset with new features
data = featureFormat(my_dataset, optimized_features_list + \
                     ['msg_to_poi_ratio', 'msg_from_poi_ratio'], \
                     sort_keys = True)
new_f_labels, new_f_features = targetFeatureSplit(data)
new_f_features = scaler.fit_transform(new_f_features)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

def evaluate_clf(grid_search, features, labels, params, iters=100):
    acc = []
    pre = []
    recall = []
    for i in range(iters):
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=i)
        grid_search.fit(features_train, labels_train)
        pred = grid_search.predict(features_test)
        acc = acc + [accuracy_score(labels_test, pred)] 
        pre = pre + [precision_score(labels_test, pred)]
        recall = recall + [recall_score(labels_test, pred)]
    print "Accuracy: {}".format(mean(acc))
    print "Precision:{}".format(mean(pre))
    #print pre
    print "Recall:{}".format(mean(recall))
    best_params=grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))




nb_clf=naive_bayes.GaussianNB()
nb_param ={}
nb_grid_search=GridSearchCV(nb_clf,nb_param)

print "\nNaive Bayes Model without new features" 
evaluate_clf(nb_grid_search,features,labels,nb_param)  

print("\nEvaluate naive bayes model with new features")
evaluate_clf(nb_grid_search, new_f_features, new_f_labels, nb_param)

## when i used k=10 the accuracy,precision,recall for the best algorithm(which I selected) got reduced
#Naive Bayes Model without new features
#Accuracy: 0.796976744186
#Precision:0.295933779353
#Recall:0.35299025974

#Evaluate naive bayes model with new features
#Accuracy: 0.801162790698
#Precision:0.312472850111
#Recall:0.370062049062

#when i used k=5 the accuracy,precision,recall for the best algorithm(which I selected) 
#Naive Bayes Model without new features
#Accuracy: 0.863571428571
#Precision:0.463333333333
#Recall:0.358541125541

#Evaluate naive bayes model with new features
#Accuracy: 0.852619047619
#Precision:0.438571428571
#Recall:0.361572150072

#when i used k=7 the accuracy,precision,recall for the best algorithm(which I selected) 
#Naive Bayes Model without new features
#Accuracy: 0.854761904762
#Precision:0.432977633478
#Recall:0.373191558442

#Evaluate naive bayes model with new features
#Accuracy: 0.842619047619
#Precision:0.395617965368
#Recall:0.37384992785

#On running tester.py for k=5 and k=7
#Accuracy: 0.85464       Precision: 0.48876      Recall: 0.38050
#Accuracy: 0.84671       Precision: 0.45660      Recall: 0.38400




from sklearn import linear_model


lo_clf = Pipeline(steps=[
        ('scaler', preprocessing.StandardScaler()),
        ('classifier', linear_model.LogisticRegression())])
         
lo_param = {'classifier__tol': [1, 0.1, 0.01, 0.001, 0.0001], \
            'classifier__C': [0.1, 0.01, 0.001, 0.0001]}
lo_grid_search = GridSearchCV(lo_clf, lo_param)
print("\nEvaluate logistic regression model without new features")
evaluate_clf(lo_grid_search, features, labels, lo_param)

lo_clf=linear_model.LogisticRegression()
print("\nEvaluate logistic regression model with new features")
evaluate_clf(lo_grid_search, new_f_features, new_f_labels, lo_param)

from sklearn.ensemble import RandomForestClassifier
rfc_clf=RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
rfc_param={}
rfc_grid_search=GridSearchCV(rfc_clf,rfc_param)
print "\nRandom Forest Classifier without new features"
evaluate_clf(rfc_grid_search,features,labels,rfc_param)

rfc_clf=RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
print "\nRandom Forest Classifier with new features"
evaluate_clf(rfc_grid_search,new_f_features, new_f_labels,rfc_param)

from sklearn.ensemble import AdaBoostClassifier
adc_clf=AdaBoostClassifier()
adc_param={}
adc_grid_search=GridSearchCV(adc_clf,adc_param)
print "\nAdaBoost Classifier without new features"
evaluate_clf(adc_grid_search,features,labels,adc_param)

adc_clf=AdaBoostClassifier()
print "\nAdaBoost Classifier with new features"
evaluate_clf(adc_grid_search,new_f_features, new_f_labels,adc_param)


from sklearn.tree import DecisionTreeClassifier
dt_clf=DecisionTreeClassifier(max_depth=5)
dt_param={'criterion':['gini','entropy'],'min_samples_split':[2,30,40,50]}
dt_grid_search=GridSearchCV(dt_clf,dt_param)
print "\nDecision Tree Classifier without new features"
evaluate_clf(dt_grid_search,features,labels,dt_param)

dt_clf=DecisionTreeClassifier(max_depth=5,min_samples_split=40)
print "\nDecision Tree Classifier with new features"
evaluate_clf(dt_grid_search,new_f_features, new_f_labels,dt_param)

from sklearn import svm
s_clf = svm.SVC()
s_param = {'kernel': ['rbf', 'linear', 'poly'], 'C': [0.1, 1, 10, 100, 1000],\
           'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'random_state': [42]}    
s_grid_search = GridSearchCV(s_clf, s_param)
print("\nEvaluate svm model without new features")
evaluate_clf(s_grid_search, features, labels, s_param)

s_clf = svm.SVC(kernel='linear',C=1,random_state=42,gamma=1)
print("\nEvaluate svm model with new features")
evaluate_clf(s_grid_search, new_f_features, new_f_labels, s_param)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split

    
#clf=RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
#clf=svm.SVC(kernel='rbf',C=0.1,random_state=42,gamma=1)
clf=naive_bayes.GaussianNB()

final_features_list = optimized_features_list 
PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, final_features_list)