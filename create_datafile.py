from sklearn.model_selection import train_test_split
from datasets import load_dataset

dataset = load_dataset("Deysi/spam-detection-dataset")

X = dataset["train"]["text"]
Y = dataset["train"]["label"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

i = 1

while i <= len(X_train) :
    with open(f'./train/email{i}.txt', 'w', encoding = 'utf-8') as file:
        file.write(X_train[i - 1])
    i += 1
    
i = 1

while i <= len(X_train) :
    with open(f'./train_label/label{i}.txt', 'w', encoding = 'utf-8') as file:
        file.write(Y_train[i - 1])
    i += 1
    
i = 1

while i <= len(X_test) :
    with open(f'./test1/email{i}.txt', 'w', encoding = 'utf-8') as file:
        file.write(X_test[i - 1])
    i += 1
    
i = 1

while i <= len(X_test) :
    with open(f'./test1_label/label{i}.txt', 'w', encoding = 'utf-8') as file:
        file.write(Y_test[i - 1])
    i += 1