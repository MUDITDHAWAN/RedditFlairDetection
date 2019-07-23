# precogflairdetection

web app link: https://predictflairprecog.herokuapp.com/
uploaded files:
datafinal.csv : final data with flairs only corresponding to r/india
datawithoutnanflair.csv  : data used for model creation 
model3.pkl : pickled model
mongodbwithpython.py : script to upload data to MongoDB
nlpnew.py : python file for training models after discovering the problem due to skewness of data
Procfile
pushshiftnew.py : script for using pushshift to scrape data from reddit for only flairs from r/india
requirements.txt 
runtime.txt
script.py : web app script
templates:- index.html
	- result.html
	-statistics.html (not complete)

References:
https://devcenter.heroku.com/articles/git
https://github.com/IIIT-Delhi/Arche
/www.w3schools.com/bootstrap/bootstrap_collapse.asp
https://www.w3schools.com/howto/howto_css_menu_icon.asp
https://www.tutorialspoint.com/flask/flask_url_building
multiple answers from https://stackoverflow.com/
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
https://blog.statsbot.co/ensemble-learning-d1dcd548e936
https://stackoverflow.com/questions/37615544/f1-score-per-class-for-multi-class-classification
https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.sparse.csr_matrix.todense.html
https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
https://towardsdatascience.com/machine-learning-multiclass-classification-with-imbalanced-data-set-29f6a177c1a
https://codepen.io/colorlib/pen/rxddKy
