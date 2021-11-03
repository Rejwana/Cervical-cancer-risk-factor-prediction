# from zipfile import ZipFile
# with ZipFile("kag_risk_factors_cervical_cancer.zip",'r') as zip:
#   zip.extractall()
#   print("Done")

import math
import numpy as np
import matplotlib.pyplot as plt
import  scipy

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from itertools import cycle
from scipy import interp

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

from autoimpute.imputations import SingleImputer, MultipleImputer
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

def loadCsvDataset(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    return dataset


def load_cervical_dataset(samplesize):
    """ A cervical cancer risk factor dataset """
    filename = "kag_risk_factors_cervical_cancer.csv"
    dataset = loadCsvDataset(filename)
    return dataset


def main():
    samplesize = 858
    dataset = load_cervical_dataset(samplesize)
    X = dataset[1:, :31]
    Y = dataset[1:, 32]

    # df = pd.DataFrame(X,Y)
    # imp = SingleImputer(df,strategy="mean")
    # X_imp = imp.fit_transform(X)

    Stratigy = ['norm', 'norm', 'norm', 'norm', 'Binomial Logistic Regression',
                                     'pmm', 'pmm', 'Binomial Logistic Regression', 'pmm','Binomial Logistic Regression','pmm','Binomial Logistic Regression',
                                     'pmm', 'Binimial Logistic Regression', 'Binimial Logistic Regression', 'Binimial Logistic Regression', 'Binimial Logistic Regression',
                                     'Binimial Logistic Regression', 'Binimial Logistic Regression', 'Binimial Logistic Regression', 'Binimial Logistic Regression',
                                     'Binimial Logistic Regression', 'Binimial Logistic Regression','Binimial Logistic Regression', 'Binimial Logistic Regression',
                                     'Binimial Logistic Regression', 'Binimial Logistic Regression', 'Binimial Logistic Regression','Binimial Logistic Regression',
                                     'Binimial Logistic Regression','Binimial Logistic Regression', 'Binimial Logistic Regression']

    imp = IterativeImputer(max_iter=10000, random_state=0)
    imp.fit(X,Y)

    IterativeImputer(add_indicator=False, estimator=None,
                        imputation_order='random', initial_strategy= Stratigy,
                        max_iter=15000, max_value=None, min_value=None,
                        missing_values=np.nan, n_nearest_features=5,
                        random_state=0, sample_posterior=False, tol=0.001,
                        verbose=0)

    X_imp = imp.transform(X)
    X = X_imp
    #df = pd.DataFrame(X, Y)
    #df.to_csv(index=False, path_or_buf='Cirvical_imp1.csv')
    '''Over sample using SMOTE() combines sampling using SMOTEomek one test for non resampled data'''

    X_resampled, Y_resampled = SMOTE().fit_resample(X, Y)
    #smt = SMOTETomek(random_state=0)
    #X_resampled, Y_resampled = smt.fit_resample(X, Y)
    X = X_resampled
    Y = Y_resampled

    clf2 = MLPClassifier(
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001, power_t=0.5, max_iter=3000)

    clf1 = LogisticRegression(random_state=0, multi_class='ovr', fit_intercept=True, class_weight='balanced')

    clf3 = svm.SVC(C=1.0, degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001,
                   cache_size=200, verbose=False, max_iter=-1, decision_function_shape='ovr',
                   random_state=None)

    parameters = {
        'Logistic Regression': {'tol': [0.0001, 0.001, 0.01, 0.1], 'max_iter': [100, 1000], 'penalty': ['l2'],
                             'solver': ['liblinear', 'lbfgs']},
        #[(4, 8, 4), (8, 16), (32)]
        'Neural Network': {'hidden_layer_sizes':[(4,8),(8,4)],
                           'activation' : ['relu','logistic', 'tanh']
                           },
        'Support Vector Machine': {
            'kernel':['rbf', 'linear', 'poly', 'sigmoid']
        }
    }

    classalgs = {
        'Logistic Regression': clf1,

        'Support Vector Machine': clf3,
        'Neural Network': clf2
    }


    n_splits = 8
    numruns = 3
    m = n_splits*numruns
    count_array = np.zeros((3, m))
    tn = np.zeros((3, m))
    fp = np.zeros((3, m))
    fn = np.zeros((3, m))
    tp = np.zeros((3, m))

    tpr = np.zeros((3, m+2))
    fpr = np.zeros((3, m+2))
    precision = np.zeros((3, m))
    recall = np.zeros((3, m))
    F1 = np.zeros((3, m))

    '''
    Run algorithms
    '''
    for r in range(numruns):
        j = 0

        for learnername, Learner in classalgs.items():
            params = parameters.get(learnername, [None])
            i = 0
            skf = StratifiedKFold(n_splits=8)
            skf.get_n_splits(X, Y)
            for train_index, test_index in skf.split(X, Y):
                Xtrain, Xtest = X[train_index], X[test_index]
                Ytrain, Ytest = Y[train_index], Y[test_index]
                #X_resampled, Y_resampled = ADASYN().fit_resample(Xtrain, Ytrain)
                #X_resampled, Y_resampled = SMOTE().fit_resample(Xtrain, Ytrain)
                grid = GridSearchCV(Learner, params, cv=8)
                grid.fit(X_resampled, Y_resampled)
                predictions = grid.best_estimator_.predict(Xtest)

                col = n_splits * r + i

                tn[j][col], fp[j][col], fn[j][col], tp[j][col] = confusion_matrix(Ytest, predictions).ravel()
                print(tn[j][col], fp[j][col], fn[j][col], tp[j][col])
                tpr [j][col] = tp[j][col]/(tp[j][col]+fn[j][col])
                fpr [j][col] = fp[j][col]/(fp[j][col]+tn[j][col])

                precision[j][col] = tp[j][col] / (tp[j][col] + fp[j][col])
                recall[j][col] = tp[j][col] / (tp[j][col] + fn[j][col])
                F1[j][col] = (2* precision[j][col] * recall[j][col]) / ( precision[j][col] + recall[j][col])
                score = accuracy_score(predictions, Ytest)

                count_array[j][col] = score
                i += 1

            j += 1
    print(count_array)

    '''
    Accuracy Score
    '''

    j=0
    accuracy = np.zeros(3)
    mean_tn = np.zeros(3)
    mean_fp = np.zeros(3)
    mean_fn = np.zeros(3)
    mean_tp = np.zeros(3)
    mean_tpr = np.zeros(3)
    mean_fpr = np.zeros(3)
    mean_precision = np.zeros(3)
    mean_recall = np.zeros(3)
    mean_F1 = np.zeros(3)
    for learnername, Learner in classalgs.items():
        accuracy[j] = np.mean(count_array[j])
        mean_tn[j] = np.mean(tn[j])
        mean_fp[j] = np.mean(fp[j])
        mean_fn[j] = np.mean(fn[j])
        mean_tp[j] = np.mean(tp[j])
        mean_tpr[j] = np.mean(tpr[j])
        mean_fpr[j] = np.mean(fpr[j])
        mean_precision[j] = np.mean(precision[j])
        mean_recall[j] = np.mean(recall[j])
        mean_F1[j] = np.mean(F1[j])


        print('Avarage Accuracy of ' + learnername + 'is:' + str(accuracy[j]))
        print('true negative of ' + learnername + 'is:' + str(mean_tn[j]))
        print('false positive of ' + learnername + 'is:' + str(mean_fp[j]))
        print('false negative of ' + learnername + 'is:' + str(mean_fn[j]))
        print('true positive of ' + learnername + 'is:' + str(mean_tp[j]))
        print('true positive rate of ' + learnername + 'is:' + str(mean_tpr[j]))
        print('false positive of ' + learnername + 'is:' + str(mean_fpr[j]))
        print('Precision of ' + learnername + 'is:' + str(mean_precision[j]))
        print('Recall of ' + learnername + 'is:' + str(mean_recall[j]))
        print('F1 score of ' + learnername + 'is:' + str(mean_F1[j]))
        lw = 2
        j += 1


    '''
    SVM vs LR
    '''
    winner1 = np.zeros((2, m))

    for i in range(m):
        if count_array[0][i] > count_array[1][i]:
            winner1[0][i] = 1
        elif count_array[0][i] < count_array[1][i]:
            winner1[1][i] = 1

    print(winner1)

    p_val = scipy.stats.binom_test(sum(winner1[1]), n=m, p=0.5)
    print(p_val)
    significance_threshold = 0.05
    if ((p_val <= significance_threshold )):
        print(
            "Quality of Support Vector Machine is better than that of Logistic Regression for our Cervical Cancer Dataset")

    else:
        print("Quality are same"
            )

    '''
    SVM vs NN
    '''
    winner2 = np.zeros((2, m))

    for i in range(m):
        if count_array[2][i] > count_array[1][i]:
            winner2[0][i] = 1
        elif count_array[2][i] < count_array[1][i]:
            winner2[1][i] = 1

    print(winner2)

    p_val = scipy.stats.binom_test(sum(winner2[1]), n=m, p=0.5)
    print(p_val)
    significance_threshold = 0.05
    if ((p_val <= significance_threshold)):
        print(
            "Quality of Support Vector Machine is better than that of Neural Network for our Cervical Cancer Dataset")

    else:
        print("Quality of SVM and NN are same")


    plt.figure()
    plt.plot(fpr[0], tpr[0], lw=lw, label='ROC curve logistic')
    plt.plot(fpr[1], tpr[1], lw=lw, label='ROC curve SVM')
    plt.plot(fpr[2], tpr[2], lw=lw, label='ROC curve NN')
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


main()


