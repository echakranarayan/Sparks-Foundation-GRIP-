#!/usr/bin/env python
# coding: utf-8

# # Task 1:
# 
# ## Predict the percentage of an student based on the number of study hours.

# #### Importing the required libraries.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Importing Data

# In[2]:


URL = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
Data = pd.read_csv(URL)
print("Data Imported Successfully")

Data.head(6)


# In[3]:


Data.shape


# In[4]:


Data.info()


# In[5]:


Data.describe()


# #### Plotting the data based on No of marks scored and No of Study Hours  

# In[6]:


Data.plot(x="Hours", y="Scores", style="*")
plt.title("Percentage Students Scored acc to the No of Hours Students Studied")
plt.xlabel("No of Hours Students Studied")
plt.ylabel("Percentage Students Scored")
plt.show()


# ### Now We've to split the data for train and test 

# In[20]:


X = Data.iloc[:, :-1].values  
y = Data.iloc[:, -1].values  


# In[31]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# In[32]:


from sklearn.linear_model import LinearRegression  
Lr = LinearRegression()  
Lr.fit(X_train, y_train) 

print("Training complete.")


# In[33]:


print("Coeffecient:",Lr.coef_)


# In[34]:


print("Intercept:",Lr.intercept_)


# #### Predicting the y values

# In[35]:


y_pred = Lr.predict(X_test)
print(y_pred)


# #### Comparing Actual Data eith Predicted Data

# In[37]:


PD = pd.DataFrame({"Actual":y_test,"Predicted":y_pred})
print(PD)


# #### Predicting What will be predicted score if a student studies for 9.25 hrs/ day

# In[38]:


Hours = 9.25
Rpred = Lr.predict([[Hours]])


# In[41]:


print("Number of Study hours= 9.25 \nPredicted Percentage=",(Rpred))


# #### Predicting the Graph with the predicted Line of Regression

# In[47]:


line = Lr.coef_*X + Lr.intercept_


# In[48]:


plt.scatter(X,y)
plt.plot(X,line)
plt.show


# In[50]:


from sklearn import metrics
print('Mean Absolute error{}'.format(metrics.mean_absolute_error(y_test,y_pred)))

