import os
import re
import numpy as np
import lda
import pandas as pd
import json
import math

# 1.get all info from part-*
data = []
for file in os.listdir(os.getcwd()):
    if re.match(r'part-*', file):
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                # template is a kind template for all null situation
                template = '#'.join([str(node) + '|0' for node in range(0, 13)])

                # first split the string
                container = (line.split(','))

                # then reform it
                # solution 1 looks elegant(use [:] instead of range), however, it works poor, because 17==18==19
                # so now this is solution 2
                for item in range(17, 21):
                    if (container[item] == '' or container[item] == '\n'):
                        container[item] = template
                # finally connect it
                recontainer = ','.join(container)
                data.append(recontainer)

# 2. change the format of the infos
template = '#'.join([str(node) + '|0' for node in range(0, 13)])
documents=[]
for tmp in data:
    result_data={}
    dict = tmp.split(',')
    result_data.update({"name":dict[0]})
    result_OD = []
    for index in range(17,21):
        OD_string = "OD"+str(index-16)+" is "
        if (dict[index]!=template and dict[index]!=template+"\n"):
            dict[index] = dict[index][12:]
            XXXX = dict[index]
            dict[index] = OD_string + dict[index]
            result_OD.append(dict[index])
    # change word to 4 OD only 2017-04-21
    if (len(result_OD)>0):
        result_data.update({"words":result_OD})
        documents.append(result_data)

# 3. get three essential part of LDA, thay're vocabulary , title and X(matrix)
vocabulary = set()
title = list()
for item in documents:
    title.append(item["name"])
    for attr in item["words"]:
        vocabulary.add(attr)
vocabulary = tuple(vocabulary)
title = tuple(title)
dim1 = len(title)
dim2 = len(vocabulary)
X = np.zeros((dim1, dim2), np.int8)

# 4. print length of them to check
print("length of documents is {}", format(len(documents)))
print("length of vocabulary is {}", format(len(vocabulary)))
print("length of title is {}", format(len(title)))
print("sum of X is {}", format(X.sum()))

# 5. initial X
# this step will take about 5 minutes, 6000w operations in all
for x in range(0, dim1):
    for y in range(0, dim2):
        if vocabulary[y] in documents[x]["words"]:
            X[x][y] = X[x][y] + 1
    print(x)
# pre work over

# 5.5 use this instead of step 5
# X = np.load('X.npy')

for num in range(2, 7, 1):
    model = lda.LDA(n_topics=num, n_iter=1500, random_state=1)
    model.fit(X)

    topic_word = model.topic_word_  # model.components_ also works
    n_top_words = 20

    # topic to word part
    # then write the result to a .txt file
    T_w = []
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocabulary)[np.argsort(topic_dist)][:-n_top_words:-1]
        T_word = 'Topic {}: {}'.format(i, ' '.join(topic_words))+'\n'
        T_rate = topic_dist[np.argsort(topic_dist)][:-n_top_words:-1]
        T_w.append(T_word)
        T_w.append(T_rate)
    f_name = "Topic_word_num=" + str(num) + ".txt"
    with open(f_name, mode='w') as f:
        for item in T_w:
            f.writelines(str(item)+'\n')

    # title to topic part
    # then write result to a .csv file
    column = ["document"]
    rows = []
    for index in range(0, num):
        column.append("topic" + str(index))

    doc_topic = model.doc_topic_
    for index, tt in enumerate(title):
        # print("{} (top topic: {})".format(tt, doc_topic[index].argmax()))
        row_of_rows = [tt]
        for index, rate in enumerate(doc_topic[index]):
            # print(index)
            # print(tmp[index])
            row_of_rows.append(rate)
        rows.append(row_of_rows)
    df = pd.DataFrame(rows, columns=column)
    f_name = "Doc_topic" + str(num) + ".csv"
    df.to_csv(f_name)

    print('---------------------------------------OVER---------------------------------------------')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
