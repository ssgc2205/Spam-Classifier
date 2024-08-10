
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os

x_train = []
y_train = []
x_test = []
y_test = []

i = 1

while i <= 6540 :
    with open(f'./train/email{i}.txt', 'r', encoding = 'utf-8') as file :
        v = file.read()
        x_train.append(v)
    i += 1
    
i = 1

while i <= 6540 :
    with open(f'./train_label/label{i}.txt', 'r', encoding = 'utf-8') as file :
        v = file.read()
        y_train.append(v)
    i += 1
    
i = 1

while i <= 1635 :
    with open(f'./test1/email{i}.txt', 'r', encoding = 'utf-8') as file :
        v = file.read()
        x_test.append(v)
    i += 1
    
i = 1

while i <= 1635 :
    with open(f'./test1_label/label{i}.txt', 'r', encoding = 'utf-8') as file :
        v = file.read()
        y_test.append(v)
    i += 1



vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(x_train)
Y_train = np.array(y_train)

X_test = vectorizer.transform(x_test)
Y_test = np.array(y_test)



def SVM_Classifier1(X_train, Y_train, X_test, Y_test) :
        
    svm = SVC(kernel = "linear")

    svm.fit(X_train, Y_train)

    Y_prediction = svm.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_prediction)

    print(f'Accuracy is : {accuracy}')
    
    print(f'Error is : {1 - accuracy}')

SVM_Classifier1(X_train, Y_train, X_test, Y_test)

def SVM_Classifier2(X_train, Y_train, X_test) :
    svm = SVC(kernel = "linear")
    svm.fit(X_train, Y_train)
    Y_prediction = svm.predict(X_test)
    
    for i, prediction in enumerate(Y_prediction) :
        print(f'Prediction for email{i + 1} is {prediction}')

# path = "test"

x_t = []

if os.path.exists("test") :
    emails = os.listdir("test")
    for email in emails :
        path = os.path.join("test", email)
        with open(path, 'r', encoding = 'utf-8') as file:
            v = file.read()
            x_t.append(v)
            
    vec = TfidfVectorizer()
    X_t = vectorizer.transform(x_t)
    SVM_Classifier2(X_train, Y_train, X_t)
    



