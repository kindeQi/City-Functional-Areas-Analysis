import json
import numpy as np
import math
import pandas as pd
import lda

# math formula function
def myNorm(x):
    y = int(math.floor(np.log(x+np.e-1))+1)
    return y

# 1.read .json from a file
d=[]
with open('POI_matrix.json') as json_data:
    d = json.load(json_data)

# 2.read all poi infos and then normalize them
# and change POI name from 1 to 21 , instead of 0 to 20
documents=[]
for i,f_dimsion in enumerate(d):
    for j,s_dimsion in enumerate(f_dimsion):
        tmp_word = []
        for index,item in enumerate(s_dimsion):
            # s_dimsion[index] = "POI" + str(index+1)+" is " +myNorm(item)
            # if ((index+1)!=12 and (index+1)!=8 and (index+1)!=9):
            POI_index = "POI" + str(index+1)
            for norm in range(1,item):
                tmp_word.append(POI_index)
            # if myNorm(item)>1:
            #     tmp_word.append(POI_index +" is "+ str(myNorm(item)))
        index = str(i)+"|"+str(j)
        # only record POI >0 part to documents
        if len(tmp_word)>0:
            tmp = {"name": index,"words": tmp_word}
            documents.append(tmp)

# 2.5 write down all the POI that the All number is 0
# f_name = "All_zero_POI"+".json"
# all_zero_json = {}
# with open (f_name,mode='w') as f:
#     for item in documents:
#         if (len(item["words"]))==0:
#             all_zero_json.update({item["name"]:"All POI are zeros"})
#             # f.writelines(item["name"] +": All are zero" + '\n')
#     json.dump(all_zero_json, f)

# 3. get three essential part of LDA, thay're vocabulary , title and X(matrix)
vocabulary = set()
title = list()
for item in documents:
    title.append(item["name"])
    for attr in item["words"]:
        vocabulary.add(attr)
vocabulary = tuple(vocabulary)
title = tuple(title)
dim1=len(title)
dim2=len(vocabulary)
X=np.zeros((dim1,dim2),np.int8)

# 4. print length of them to check
print ("length of documents is {}",format(len(documents)))
print ("length of vocabulary is {}",format(len(vocabulary)))
print ("length of title is {}",format(len(title)))
print ("sum of X is {}",format(X.sum()))

# 5. initial X
# this step will take about 5 minutes, 6000w operations in all
for x in range(0,dim1):
    for y in range(0,dim2):
        if vocabulary[y] in documents[x]["words"]:
            X[x][y] = X[x][y]+1
    print(x)
# pre work over

# 5.5 use this instead of step 5
# X = np.load('X.npy')

for num in range(3,7,1):

    model = lda.LDA(n_topics=num, n_iter=1500, alpha=0.33, random_state=1)
    model.fit(X)

    topic_word = model.topic_word_  # model.components_ also works
    n_top_words = 20

    # topic to word part
    # then write the result to a .txt file
    T_w = []
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocabulary)[np.argsort(topic_dist)][:-n_top_words:-1]
        T_word = 'Topic {}: {}'.format(i, ' '.join(topic_words))
        T_rate = topic_dist[np.argsort(topic_dist)][:-n_top_words:-1]
        T_w.append(T_word)
        T_w.append(T_rate)
    f_name = "Topic_word_num="+str(num)+".txt"
    with open (f_name,mode='w') as f:
        for item in T_w:
            f.writelines(str(item)+'\n')

    # title to topic part
    # then write result to a .csv file
    column = ["document"]
    rows = []
    for index in range(0,num):
        column.append("topic"+str(index))

    doc_topic = model.doc_topic_
    for index,tt in enumerate(title):
        # print("{} (top topic: {})".format(tt, doc_topic[index].argmax()))
        row_of_rows = [tt]
        for index,rate in enumerate(doc_topic[index]):
            # print(index)
            # print(tmp[index])
            row_of_rows.append(rate)
        rows.append(row_of_rows)
    df = pd.DataFrame(rows,columns=column)
    f_name = "Doc_topic" + str(num) + ".csv"
    df.to_csv(f_name)

    print ('---------------------------------------OVER---------------------------------------------')
    print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
