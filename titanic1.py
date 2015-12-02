# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.metrics.classification import accuracy_score


train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')
predictorsLabel = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Fill in na and transfer non_value columns
def harmonizeData(titanic):
    
    titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
    
    titanic.loc[titanic['Sex']=='male', 'Sex'] = 0
    titanic.loc[titanic['Sex']=='female', 'Sex'] = 1
    
    titanic['Embarked'] = titanic['Embarked'].fillna('S')
    titanic.loc[titanic['Embarked']=='S', 'Embarked'] = 0
    titanic.loc[titanic['Embarked']=='C', 'Embarked'] = 1
    titanic.loc[titanic['Embarked']=='Q', 'Embarked'] = 2
    
    titanic['Fare'] = titanic['Fare'].fillna(titanic['Fare'].median())
    
    return titanic


"""
# def addFeature(titanic):
#     
#     return titanic 
"""  

 
def createSubmission(alg, trainData, testData, predictors, filename):
    
    alg.fit(trainData[predictors], trainData['Survived'])
    predictions = alg.predict(testData[predictors])
    
    submission = pd.DataFrame({'PassengerId': testData['PassengerId'],
                                'Survived': predictions})
    
    submission.to_csv(filename, index=False)
    
 
# Below will test different algorithms to get an intuitive evaluation
# They are: Logistic Regression, RandomForest, GBDT and SVM 
# In last, try an ensemble method on combining two different algorithm 
# 0.78788 --0.75120
def predictByLogisticRegres(trainData, predictors=predictorsLabel):
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_validation import cross_val_score
    
    alg = LogisticRegression(random_state=1)
    scores = cross_val_score(alg, trainData[predictors], 
                             trainData['Survived'], cv=3)
    
    
    print 'the score by Logistic Regression is: ', scores.mean()

# 0.82379 --0.76077
def predictByRandomForest(trainData, predictors=predictorsLabel):
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cross_validation import cross_val_score
    
    alg = RandomForestClassifier(random_state=1, n_estimators=150,
                                 min_samples_split=6, min_samples_leaf=3)
    scores = cross_val_score(alg, trainData[predictors],
                             trainData['Survived'], cv=3)
    
    print 'The score by Random Forest is: ', scores.mean()
    

# 0.81936 -- 0.77990
def predictByGradientBoosting(trainData, predictors=predictorsLabel):
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.cross_validation import cross_val_score
    
    alg = GradientBoostingClassifier(random_state=1, n_estimators=25,
                                     max_depth=3)
    scores = cross_val_score(alg, trainData[predictors],
                             trainData['Survived'], cv=5)
    
    print 'The score by GBDT is: ', scores.mean()
    

# 0.79580  --0.77033
def predictBySVM(trainData, predictors=predictorsLabel):  
    
    from sklearn.svm import SVC
    from sklearn.cross_validation import cross_val_score
    
    # After parameter tuning, as a result, C=500, gamma=0.001 are prefered
    alg = SVC(random_state=1, kernel='rbf', gamma=0.001, C=500)
    scores = cross_val_score(alg, trainData[predictors], 
                             trainData['Survived'], cv=5)
    
    print 'The score by SVM paramatered by C=500 gamma=0.001 is: ', scores.mean()


"""
# Different voting methods
def getEnsembleResult(PredictionsList, method='averagingProbs'):
    if method == 'averagingProbs':
        return (PredictionsList[0]+PredictionsList[1])/2
    elif method == 'majorityVoting':
        return  ***
    else: 
        print 'Only "majorityVoting" or "averagingProbs" method can be choosed.'
"""      


# Think: How to apply this ensemble method to the whole traing set?
# 0.80920 --0.78469
def predictByEnsemble(trainData, 
                      predictors1=predictorsLabel, 
                      predictors2=predictorsLabel):
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.cross_validation import KFold
    
    kf = KFold(trainData.shape[0], n_folds=5, random_state=1)
    algorithms = [[GradientBoostingClassifier(random_state=1, n_estimators=25,max_depth=3),predictors1],
                  [SVC(random_state=1, kernel='rbf', gamma=0.001, C=500, probability=True), predictors2]]
        
    predictions = []
    for train, test in kf:
        
        trainTarget = trainData['Survived'].iloc[train]
        # Store predictions of these two algorithms separately
        bothPredictions = []
        for alg, predictors in algorithms:
            
            alg.fit(trainData[predictors].iloc[train,:],trainTarget)
            # 要使用.predict_proba()得到各个类的概率; 因此，SVC中的probability=True
            # .astype(float)is necessary.
            testPredictions = alg.predict_proba(trainData[predictors].iloc[test,:].astype(float))[:,1]
            bothPredictions.append(testPredictions)
            
        # Choose a method among 'majority voting' or 'averaging probability'
        testPredictions = (bothPredictions[0]+bothPredictions[1])/2
        testPredictions[testPredictions<=.5] = 0
        testPredictions[testPredictions>.5] = 1
        predictions.append(testPredictions)

    # concatenate the seperate predictions
    predictions = np.concatenate(predictions, axis=0)
    accuracy = sum(predictions[predictions == trainData['Survived']])/len(predictions)
    
    print 'The accuracy by ensemble of GBDT&SVM is: ', accuracy

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    