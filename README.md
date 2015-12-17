# Kaggle Competition Practice :: 00-titanic

This post is part of my first try for [#Kaggle Competitions](https://www.kaggle.com/competitions), 
it's really a classical routine to acquire a comprehensive understanding about how to address problem in reality. 

---

## Introduction
First, preview the 'train.csv' file to get an intuitive understanding for the data info.
Obviously, 'Sex', 'Age', 'Pclass' should be important to decide whether one will survive(which 
can be quickly operated in excel throughing pivot table, that don't need really a numerical value).

Then, read the file by pandas.read_csv and by using .describe() method we can find there'are missing
values! So how to tackle about it? 1) Fill the value by median value 2) Note that: we must apply the 
same tunning or scaling to 'test.csv', which means, we can define a function, named harmoizeData, to 
operator our datas 3) Remember to transfer nonnumerical values, cause classifier needs numerical input.

I try nearly all the classifier algorithms, so as to get an comprehensive comparision on the 
result. Concretly, several aspects are worth mentioned.
- Thankfully to us, after a period of writing 
my own codes for different ML algorithms and deducing & understanding principles, now i found it's easy 
to utlize the powerful scikit learn module: [sklearn](http://scikit-learn.org/stable/index.html) to generalize
a much easier and quicker predictable module. Of course, it's must be just a start, a lot details are
supposed to know and dis/advantages to handle. BTW, this blog maybe helpful: [Machine Learning Mastery](http://machinelearningmastery.com/blog). And greatly recommender this book: [「统计学习方法」](http://book.douban.com/subject/10590856/).
- 'LinearRegression'-->'LogisticalRegression'-->'RandomForest-
Classifier'-->'GradientBoostingClassifer'(GBDT)-->'svm.SVC'-->'[ensemble](http://www.scholarpedia.org/article/Ensemble_learning)
on "GBDT"&"SVM"' are successively tried before focusing on feature engineering. 
- About [#feature engineering](https://getpocket.com/a/read/723691867)
- About [#cross_validation](http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)
- About [#Ensemble methods](http://www.scholarpedia.org/article/Ensemble_learning)

