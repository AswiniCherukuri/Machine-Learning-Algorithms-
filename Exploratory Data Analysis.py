
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[88]:


#Read train data set
df_train = pd.read_csv(r"E:\Kaggle Problems\Titanic problem\Titanic data set\train.csv")


# ## Exploratory Data Analysis

# In[89]:


df_train.head()


# In[90]:


df_train.isnull()


# In[91]:


df_train.isnull().sum()


# ### Visualize data to check null values(NAN)

# In[92]:


sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap="viridis")


# By this heatmap, we can tell that Age and Cabin columns have missing values. Out of two, Cabin column has more null values.

# In[93]:


sns.set_style("whitegrid")
sns.countplot(x="Survived", data=df_train)


# In[94]:


sns.countplot(x="Survived",hue="Sex",data=df_train)


# In[95]:


sns.countplot(x="Survived",hue="Pclass",data=df_train)


# By the above plot, we can tell that survival count of Pclass 1 is more than 2 & 3 (People who belongs to Pclass 1 are survived more).

# In[96]:


sns.distplot(df_train["Age"].dropna(),kde=False,color="green",bins=40)


# By this plot, we can understand that average age lies between 20-30. 55 is the highest age. 2-3 is the lowest age.

# In[97]:


df_train["Age"].hist(bins=40,color="green",alpha=0.3)


# In[98]:


sns.countplot(x="SibSp",data=df_train)


# By this plot, we can tell that most of the people have no siblings and spouse.

# In[99]:


df_train["Fare"].hist(bins=40,color="green",figsize=(8,5))


# In[100]:


plt.figure(figsize=(9,6))
sns.boxplot(x="Pclass",y="Age",data=df_train,color="red",palette="rainbow")


# Average age of people who belongs to Pclass 1 is 37 where as Pclass 2 is 29 and Pclass 3 is 24.
# That means, wealthier persons in the higher class tend to be older.

# ### Data Cleaning

# Let us do imputation to fill in missing values

# In[101]:


def imputeAge(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        if Pclass==2:
            return 29
        if Pclass==3:
            return 24
    else:
        return Age


# Now apply that function

# In[102]:


df_train["Age"] = df_train[["Age","Pclass"]].apply(imputeAge,axis=1)


# Now check heatmap again

# In[103]:


sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap="viridis")


# Age column is filled and no more null values !

# In[104]:


df_train.drop("Cabin",axis=1,inplace=True)


# In[105]:


df_train.head()


# In[106]:


sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap="viridis")


# Great! No null values are present

# Let's move ahead

# ### Converting Categorical Features

# We need to convert categorial features data in to integers using dummy variables. Otherwise our model won't work for categorical data inputs

# Categorical Features:
# 1. Name
# 2. Sex
# 3. Embarked

# In[107]:


pd.get_dummies(df_train["Embarked"],drop_first=True).head()


# In[108]:


Sex = pd.get_dummies(df_train["Sex"],drop_first=True)
Embarked = pd.get_dummies(df_train["Embarked"],drop_first=True)


# In[109]:


df_train.drop(["Name","Ticket","Sex","Embarked"],axis=1,inplace=True)


# In[110]:


df_train.head()


# In[111]:


# Add dummy columns
df_train = pd.concat([df_train,Sex,Embarked],axis=1)


# In[112]:


df_train.head()


# Great! Our data is ready for model !!!

# ## Building a Logistic model

# Here survived column is y variable.

# In[114]:


df_train['Survived'].head()


# Split entire data in to train and test data.

# In[119]:


from sklearn.model_selection import train_test_split


# In[120]:


X_train, X_test, Y_train, Y_test = train_test_split(df_train.drop("Survived",axis=1),
                                                    df["Survived"],test_size=0.30,
                                                    random_state=101)


# ## Training and Predicting

# In[121]:


from sklearn.linear_model import LogisticRegression


# In[122]:


logModel = LogisticRegression()
logModel.fit(X_train,Y_train )


# In[123]:


predictions = logModel.predict(X_test)


# In[132]:


predictions


# In[126]:


from sklearn.metrics import confusion_matrix


# In[127]:


accuracy = confusion_matrix(Y_test,predictions)


# In[128]:


accuracy


# In[129]:


from sklearn.metrics import accuracy_score


# In[130]:


accuracy = accuracy_score(Y_test,predictions)


# In[131]:


accuracy

