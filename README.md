# Fake-news-Detection-using-Ensemble-Learning-
Introduction:

	Do you trust all the news you hear from social media? Fake news and hoaxes have been there since before the advent of the Internet.  So how will you detect the fake news? The answer is Machine Learning. By practicing this project of detecting fake news, we will easily make a difference between real and fake news.
Following the advent of the internet, more and more users began using the web as their primary source of information and news, as it is more convenient and faster. The development, however, came with a redefined concept of fake news as content publishers began using what has come to be commonly referred to as a clickbait. Users continue to deal with sites containing false information and whose involvement tends to affect the reader’s ability to engage with actual news. The main aim of the project is to reveal the benefits of Machine Learning methodology used in the detection of fake news and their success levels in this particular application. As a result of the study, it was concluded that the success level of the project is over 90%.

Proposed Solution:

	This project of detecting fake news deals with fake and real news. Using sklearn, we’ll build a TfidfVectorizer on our dataset. Then, we’ll initialize a PassiveAggressive Classifier, Multinomial Naive Bayes Classifier, Random Forest Classifier, Support Vector Machine Classifier   and fit the model. And then we make an ensemble learning model with a hard voting and all the above mentioned classifiers. And check the accuracy score of the project.

Methodology:

Retrieving Data: The dataset we’ll use for this python project- we’ll call it news.csv , True.csv, Fake.csv, All the news bundle that was found at Kaggle.

Data Preparation: In this step we prepare the data to use it in the model the dataset had a couple of unwanted columns that were not required by the madel so we removed those columns and also filled those columns that have null values in them. Add a label to real and fake news so that we can train this data in the future.

Data Exploration: In this step we explored the data that we are going to use in our model. We tokenized the data using TFIdf Tokenizer and removed unnecessary most occurring words.
What is a TfidfVectorizer?
TF (Term Frequency): The number of times a word appears in a document is its Term Frequency. A higher value means a term appears more often than others, and so, the document is a good match when the term is part of the search terms.
IDF (Inverse Document Frequency): Words that occur many times in a document, but also occur many times in many others, may be irrelevant. IDF is a measure of how significant a term is in the entire corpus.

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


Data Modeling: In this stage we make our model and fit the data and train the data. The model we used in this project is.
We split the data into 80% training and 20% testing ratio
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=7)

Passive Aggressive Classifier
		Passive Aggressive algorithms are online learning algorithms. Such an algorithm remains passive for a correct classification outcome, and turns aggressive in the event of a miscalculation, updating and adjusting. Unlike most other algorithms, it does not converge. Its purpose is to make updates that correct the loss, causing very little change in the norm of the weight vector.

Model initialize:
PAC=PassiveAggressiveClassifier(max_iter=1000)

Multinomial Naive Bayes
		Naive Bayes Classifier Algorithm is a family of probabilistic algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of a feature.

Model initialize:
MultNB = MultinomialNB()

Random Forest Classifier
		The random forest is a classification algorithm consisting of many decision trees. It uses bagging and feature randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree.

Model initialize:
rfc=RandomForestClassifier(n_estimators= 10, random_state= 7)

Ensemble Learning Model
Then we created an ensemble model with a voting classifier containing the above models with a hard voting which follows the majority rule during predictions.

Model initialize:
Ensemb = VotingClassifier( estimators = [('PAC',PAC),('MultNB',MultNB),('rfc',rfc)], voting = 'hard')
    Ensemb.fit(tfidf_train,y_train)
    Ensemb_pred=Ensemb.predict(tfidf_test)


Result Analysis:
The accuracy of the model that we created is found to be 96 %
