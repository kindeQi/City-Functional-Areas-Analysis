import json
import numpy as np
import math

# x =int(math.floor(np.log(np.e))+0.1)
# print (x)

# math formula function
def myNorm(x):
    y = int(math.floor(np.log(x+np.e-1))+1)
    return y

# 1.read .json from a file
d=[]
with open('POI_matrix.json') as json_data:
    d = json.load(json_data)

# 2.read all poi infos and then normalize them
poi = {}
for i,f_dimsion in enumerate(d):
    for j,s_dimsion in enumerate(f_dimsion):
        for index,item in enumerate(s_dimsion):
            s_dimsion[index] = myNorm(item)
        index = str(i)+"|"+str(j)
        tmp = {index : s_dimsion}
        poi.update(tmp)

print('s')