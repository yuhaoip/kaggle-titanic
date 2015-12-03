# coding: utf-8

import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt


train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')
predictorsLabel = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Fill in na and transfer non_value columns
def harmonizeData(titanic):
    
    titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
    
    titanic.loc[titanic['Sex']=='male', 'Sex'] = 0
    titanic.loc[titanic['Sex']=='female', 'Sex'] = 1
    
    # Calculate the mode
    mode = titanic['Embarked'].dropna().mode()[0]
    titanic['Embarked'] = titanic['Embarked'].fillna(mode)
    # Convert to dummy varibles, return pd.core.series.Series object
    tempDummies = pd.get_dummies(titanic['Embarked'])
    tempDummies = tempDummies.rename(columns= lambda x: 'Embarked_'+str(x))
    titanic = pd.concat([titanic, tempDummies], axis=1)
    
    titanic['Fare'] = titanic['Fare'].fillna(titanic['Fare'].median())
    
    return titanic


# Extract title from a certain name and mapped to a id
def extractTitle(name):
    import re
    # Extract the title by regular expression.
    titleSearch = re.search(r'([A-Za-z]+)\.', name)
    if titleSearch:
        title = titleSearch.group(1)
        return title  
    return ""
      
# Combine lastName with familysize to get an unique Id 
familyIdMapping = {}
def getFamilyId(row):
    
    lastName = row['Name'].split(',')[0]
    familyId = '{0}{1}'.format(lastName, row['FamilySize'])
    # Map the id to number
    if familyId not in familyIdMapping:
        if len(familyIdMapping)==0:
            currentId = 1
        else: # Get the maximun id and add 1
            currentId = (max(familyIdMapping.items(), 
                            key=operator.itemgetter(1))[1]+1)
        familyIdMapping[familyId]=currentId
    
    return familyIdMapping[familyId]
         
    
# Except for those existing features, try to build feature engineering
def addFeatures(titanic):
    # The title such as 'Countness.''Dr.' shows the identify. Should be useful.
    # Map the titles into classified numerical value( combining the testData titles)
    titles = titanic['Name'].apply(extractTitle)
#     print 'The titels count: ', pd.value_counts(titles)
    titleMapping = {'Mr':1, 'Miss':2, 'Ms':2, 'Mrs':3, 'Master':4, 'Dr':5,
                  'Rev':6, 'Col':7, 'Major':7, 'Mlle':8, 'Mme':8,
                  'Don':9, 'Lady':10, 'Countess':10, 'Jonkheer':10,
                  'Sir':9, 'Capt':7, 'Dona':10}
    
    for k, v in titleMapping.items(): 
        titles[titles==k] = v
#     print 'The titles count: ', pd.value_counts(titles)
    titanic['Titles'] = titles
    
    # Combine last name and family size to get a unique id;
    # There are too many family ids, so compress all of the familyies
    # under 3 members into one code.
    titanic['FamilySize'] = titanic['SibSp']+titanic['Parch']
    familyIds = titanic.apply(getFamilyId, axis=1)
    familyIds[titanic['FamilySize']<2]=-1
    titanic['FamilyId'] = familyIds
    
    # Now, think about the female prop in a family
    
    return titanic 


# Choose the best K features
def chooseBestFeatures(trainData, predictors=predictorsLabel, k=5):
    
    from sklearn.feature_selection import SelectKBest, f_classif
    
    selector = SelectKBest(f_classif, k)
    selector.fit(trainData[predictors], trainData['Survived'])
    scores = selector.scores_
    print 'The scores are: ', scores
    
    # plot the feature-score bar to inspect
    plt.bar(range(len(predictors)), scores)
    plt.xticks(range(len(predictors)), predictors, rotation='vertical')
    plt.show()
    
    

def createSubmission(algorithms, trainData, testData, filename):
    
    predictions = []
    for alg, predictors in algorithms:
        alg.fit(trainData[predictors], trainData['Survived'])
        test_predictions = alg.predict_proba(testData[predictors].astype(float))[:,1]
        predictions.append(test_predictions)
    
    sumPredictions = np.array([0.0]*testData.shape[0])
    for eachItem in predictions: 
        sumPredictions += eachItem
    
    predictions = sumPredictions/len(predictions)
    predictions[predictions<=.5] = 0
    predictions[predictions>.5] = 1
    
    # Transfer np.ndarray to pd.core.series.Series
    predictions = pd.Series(predictions)
    # According to pre-insight in excel, 
    # set['Pclass'=1&'Sex'=1 to 1, 'Pclass'=2&'Sex'=0&'Age'>18 to 0]
    predictions[(testData['Pclass']==1) & (testData['Sex']==1)] = 1
    predictions[(testData['Pclass']==2) & (testData['Sex']==0) & (testData['Age']>18)] = 0
    predictions = predictions.astype(int)
    
    submission = pd.DataFrame({'PassengerId': testData['PassengerId'],
                                'Survived': predictions})
     
    submission.to_csv(filename, index=False)
        
        
        
        
# Below will test different algorithms to get an intuitive evaluation
# They are: Logistic Regression, RandomForest, GBDT and SVM 
# In last, try an ensemble method on combining two different algorithm 
# 0.78788 --0.75120; 0.78563
def predictByLogisticRegres(trainData, predictors=predictorsLabel):
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_validation import cross_val_score
    
    alg = LogisticRegression(random_state=1)
    scores = cross_val_score(alg, trainData[predictors], 
                             trainData['Survived'], cv=3)
    
    
    print 'the score by Logistic Regression is: ', scores.mean()


# 0.82379 --0.76077; 0.81818
def predictByRandomForest(trainData, predictors=predictorsLabel):
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cross_validation import cross_val_score
    
    alg = RandomForestClassifier(random_state=1, n_estimators=150,
                                 min_samples_split=6, min_samples_leaf=3)
    scores = cross_val_score(alg, trainData[predictors],
                             trainData['Survived'], cv=3)
    
    print 'The score by Random Forest is: ', scores.mean()
    

# 0.81936 -- 0.77990; 0.81817
def predictByGradientBoosting(trainData, predictors=predictorsLabel):
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.cross_validation import cross_val_score
    
    alg = GradientBoostingClassifier(random_state=1, n_estimators=25,
                                     max_depth=3)
    scores = cross_val_score(alg, trainData[predictors],
                             trainData['Survived'], cv=5)
    
    print 'The score by GBDT is: ', scores.mean()
    

# 0.79580  --0.77033; 0.80472
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
# 0.83053 --0.79904(predictors=['Pclass','Sex','Titles','FamilyId','Fare'])
# 0.83389 --(predictors=['Pclass','Sex','Titles','FamilyId','Fare','Embarked'])
# 0.81033 --(predictors=['Pclass','Sex','Titles','FamilyId','Fare','Embarked','Age'])
# 0.83165 -- featured 'Embarked' by dummy_varibles.
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
            # here, need .proba() method to caculate probability.
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

 
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    