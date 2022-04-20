from cProfile import label
import nltk as nk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import CountVectorizer as cv
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn.metrics import accuracy_score

df = pd.read_csv("spam.csv")
df_sms = df.drop(["Unnamed: 2" , "Unnamed: 3" , "Unnamed: 4"], axis = 1)
df_sms = df_sms.rename(columns = {"v1" : "label" , "v2" : "sms"})
df_sms["length"] = df_sms["sms"].apply(len)
#df_sms["label"] = df_sms["label"].map({"ham" : 0 , "spam" : 1})
df_sms.loc[: , "label"] = df_sms.label.map({"ham" : 0 , "spam" : 1})

x_train, x_test, y_train, y_test = tts(
    df_sms["sms"] , df_sms["label"] ,
    test_size = 0.20 , random_state = 1
)

count_vector = cv()

training_data = count_vector.fit_transform(x_train)
testing_data = count_vector.transform(x_test)

multi_nb = mnb(alpha = 1.0 , class_prior = None , fit_prior = True)
multi_nb.fit(training_data , y_train)

prediction = multi_nb.predict(testing_data)
print("accuracy_score : " , accuracy_score(y_test , prediction))

con_inp = input("Enter a message : ")
inp = np.array(con_inp)
inp = np.reshape(inp , (1, -1))
inp_conv = count_vector.transform(inp.ravel())

result = multi_nb.predict(inp_conv)

for i in result :
    if i == 0 :
        print("The message is not a spam")
    else :
        print("The message is a spam")