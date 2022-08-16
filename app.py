import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from slickml.classification import XGBoostCVClassifier
from slickml.optimization import XGBoostClassifierBayesianOpt
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import recall_score
#from sklearn.metrics import f1_score

st.title('Machine Learning Project')

st.write("""
# Choosing Between Datasets and Classifier
Which one is the better?
""")

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris','Digits', 'Breast Cancer' )
)

st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest', 'XGBoost')
)

def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Digits':
        data = datasets.load_digits()

    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif clf_name == 'Random Forest':
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
        learning_rate = st.sidebar.select_slider('learning_rate', options=np.linspace(0,2,num=20))
        params['learning_rate'] = learning_rate
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif  clf_name == 'Random Forest':
            clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=42)    
    else:
        clf = clf = XGBClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'],learning_rate=params['learning_rate'], random_state=42)
    return clf

clf = get_classifier(classifier_name, params)
#### CLASSIFICATION ####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)
#st.write (f'Precision =', precision_score(y_test, y_pred))
#st.write (f'Recall =', recall_score(y_test, y_pred))
#st.write (f'F1_score =', f1_score(y_test, y_pred))



#### Evaluation ####



#st.write (classification_report(y_test, y_pred))




fig = plt.figure()
cmx = confusion_matrix(y_test, y_pred)
a=sns.heatmap(cmx, square= True, annot= True)
a.set_title("Confusion Matrix")
a.set_xlabel("Actual Class")
a.set_ylabel("Predicted Class")
st.pyplot(fig)
