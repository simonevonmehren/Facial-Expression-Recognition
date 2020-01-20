
"""
Created on Tue Jan  7 10:02:31 2020

@author: Alma Fazlagiz, Simone von Mehren, Natasha Norsker
"""

import pandas as pd


#-----CREATE DATA SET-----
#downloading and reading the "train.csv" from Kaggle:

data = pd.read_csv("train.csv",sep=",")

#creating a dataframe excluding the emotion "disgust"
new_data = data.loc[(data["emotion"]==0) | (data["emotion"]==2) | (data["emotion"]==3) | (data["emotion"]==4) | (data["emotion"]==5) | (data["emotion"]==6)]

#------SPLITTING INTO TRAIN AND TEST------
#1. Making sure that there are 3171 images in every category
#2. 29/30 of 3171 goes to training and 1/30 goes to testing

#-----ANGRY------
angry_all = new_data.loc[new_data["emotion"]==0]
angry = angry_all.iloc[824:,]

angry_train = angry.iloc[106:,]
angry_test = angry.iloc[3065:,]

#-----FEAR------
fear_all = new_data.loc[new_data["emotion"]==2]
fear = fear_all.iloc[926:,]

fear_train = fear.iloc[106:,]
fear_test = fear.iloc[3065:,]

#-----HAPPY------
happy_all = new_data.loc[new_data["emotion"]==3]
happy = happy_all.iloc[4044:,]

happy_train = happy.iloc[106:,]
happy_test = happy.iloc[3065:,]

#-----SAD------
sad_all = new_data.loc[new_data["emotion"]==4]
sad = sad_all.iloc[1659:,]

sad_train = sad.iloc[106:,]
sad_test = sad.iloc[3065:,]

#-----SURPRISE------
surprise = new_data.loc[new_data["emotion"]==5]

surprise_train = surprise.iloc[106:,]
surprise_test = surprise.iloc[3065:,]


#-----NEUTRAL------
neutral_all = new_data.loc[new_data["emotion"]==6]
neutral = neutral_all.iloc[1794:,]

neutral_train = neutral.iloc[106:,]
neutral_test = neutral.iloc[3065:,]

#------CREATE TRAIN DF AND CSV-----------
frames_train = [angry_train,fear_train,happy_train,sad_train,surprise_train,neutral_train]
train_data = pd.concat(frames_train)

#replace all 6 with 1, so that the orders remains ascending
train_data = train_data.replace(6,1)

train_csv = train_data.to_csv("train_data.csv",index=False)



#------CREATE TEST DF AND CSV-----------
frames_test = [angry_test,fear_test,happy_test,sad_test,surprise_test,neutral_test]
test_data = pd.concat(frames_test)

test_data = test_data.replace(6,1)


test_csv = test_data.to_csv("test_data.csv",index=False)

#------CREATE TEST CSV FOR 73 SPECIFIC IMAGES USED FOR EXPERIMENT A---

#list of indices
list1= [49,316,237,190,437,220,480,396,526,345,456,84,626,580,567,341,599,270,125,121,622,110,517,6,570,265,211,143,407,467,98,430,217,345,534,294,271,182,148,391,487,143,505,168,545,137,190,163,2,34,571,62,351,498,284,582,372,548,602,626,579,464,333,43,224,140,90,168,450,147,368,95,369]


test_data = test_data.reset_index(drop=True)
test73 = test_data.ix[list1]

test73_csv =  test73.to_csv("test73.csv",index=False)



