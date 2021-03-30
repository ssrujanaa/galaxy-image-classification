#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/local-work/srujana/galaxy-image-classification2/Dataset.csv")
X = df['GalaxyID']
y = df['Label']
new_X = []
for i in X:
    if 'Class0' in i:
        new_X.append("Class0_" + i + '.jpg')
    elif 'Class1' in i:
        new_X.append("Class1_" + i + '.jpg')
    elif 'Class2' in i:
        new_X.append("Class2_" + i + '.jpg')
    elif 'Class3' in i:
        new_X.append("Class3_" + i + '.jpg')
    elif 'Class4' in i:
        new_X.append("Class4_" + i + '.jpg')
 
X_train1, X_test, y_train1, y_test = train_test_split(new_X, y, test_size=0.2, random_state=42, stratify = y)
X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.3, random_state=42, stratify = y_train1)

new_df = pd.DataFrame(columns = ['Data', 'Label'])
new_df.assign(Data=X_train,Label=y_train)

new_df.to_csv('/local-work/srujana/galaxy-image-classification2/Training_Data.csv')

new_df1 = pd.DataFrame(columns = ['Data', 'Label'])
new_df1.assign(Data=X_test,Label=y_test)

new_df1.to_csv('/local-work/srujana/galaxy-image-classification2/Test_Data.csv')

new_df2 = pd.DataFrame(columns = ['Data', 'Label'])
new_df2.assign(Data=X_val,Label=y_val)

new_df2.to_csv('/local-work/srujana/galaxy-image-classification2/Val_Data.csv')


# In[ ]:




