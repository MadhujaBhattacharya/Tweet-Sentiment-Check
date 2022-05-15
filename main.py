import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df= pd.read_csv('sentiment_data.csv')
df["formatted"] = (
    df["message"]
    .str.replace(r"^https?:\/\/.*[\r\n]*", "", regex=True)
    .replace(r"@[A-Za-z0-9_]+", " ", regex=True)
    .replace(r"#[A-Za-z0-9_]+", " ", regex=True)
)
x_train, x_test, y_train, y_test = train_test_split(df["formatted"],df["sentiment"],test_size=0.2, random_state=1)
count_vector= CountVectorizer()
training_data= count_vector.fit_transform(x_train)  
testing_data= count_vector.transform(x_test)
model= LogisticRegression(random_state=0, max_iter=10000)
model.fit(training_data, y_train)
prediction= model.predict(testing_data)
print("Accuracy score: {}".format(accuracy_score(y_test, prediction))) 

msg_input= input("Enter a message ")
inp= np.array(msg_input) 
inp= np.reshape(inp, (1, -1)) 
inp_count= count_vector.transform(inp.ravel())
result= model.predict(inp_count)
for i in result:
    if result[0]==0:
        print("Neutral")
    elif result[0]==1:
        print("Positive")
    elif result[0]==-1:
        print("Negative")
    else:
        print("Something went wrong!")
    
