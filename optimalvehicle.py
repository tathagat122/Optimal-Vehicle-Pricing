# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## 1. Import Libraries

# %%
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV 

# %% [markdown]
# ## 2. Load Data

# %%
df = pd.read_csv("Data_OptimalVehiclePricing.csv")

# %% [markdown]
# ## 3. Understanding the data

# %%
df.info()   


# %%
df.describe()  


# %%
df.head()


# %%
num_col = df.select_dtypes(include=np.number).columns           #numerical variables
print("Numerical columns: \n",num_col)

cat_col = df.select_dtypes(exclude=np.number).columns           #categorical variables
print("Categorical columns: \n",cat_col)

# %% [markdown]
# ## 4. Data Pre-processing

# %%
df.drop(['make'],axis=1,inplace=True)

cat = ['fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem']

df = pd.get_dummies(df,cat,drop_first=True)

# %% [markdown]
# ## 5. Exploratory Data Analysis

# %%
# Let's check the distribution of y variable
plt.figure(figsize=(8,8), dpi= 50)
sns.boxplot(df['price'])
plt.title('Price Box Plot')
plt.show()


# %%
plt.figure(figsize=(8,8))
plt.title('Price Distribution Plot')
sns.distplot(df['price'])


# %%
# Let's check the multicollinearity of features by checking the correlation matric

plt.figure(figsize=(15,15))
p=sns.heatmap(df[num_col].corr(), annot=True,cmap='RdYlGn',center=0) 

# %% [markdown]
# ## 6. Model Building

# %%
# Train test split
X = df.drop(['price'], axis = 1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=500)


# %%
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# %% [markdown]
# ## Gradient Boosting Regression
# 
# - learning_rate = 0.05

# %%
gbr = GradientBoostingRegressor(learning_rate = 0.05, random_state = 100)
gbr.fit(X_train,y_train)
y_pred = gbr.predict(X_test)

print("r2 score : ",r2_score(y_test,y_pred))
print("MAPE     : ",mean_absolute_percentage_error(y_test,y_pred))

# %% [markdown]
# **Summary :**
# 
# MAPE is quite higher, so let's try to tune the parameter again
# %% [markdown]
# ## Gradient Boosting Regression
# 
# - learning_rate = 0.1

# %%
gbr = GradientBoostingRegressor(learning_rate = 0.1, random_state = 100)
gbr.fit(X_train,y_train)
y_pred = gbr.predict(X_test)

print("r2 score : ",r2_score(y_test,y_pred))
print("MAPE     : ",mean_absolute_percentage_error(y_test,y_pred))

# %% [markdown]
# **Summary :**
# 
# MAPE has improved as compared to the earlier model, let's try to tune the parameter using gridsearch
# %% [markdown]
# ## Grid Search
# 

# %%
gbr = GradientBoostingRegressor(random_state = 100)

# defining parameter range 
param_grid={'n_estimators':[100,500,1000], 
            'learning_rate': [0.2,0.15, 0.1],
            'max_depth':[2,3,4,6], 
            'min_samples_leaf':[1,3,5]}   
  
grid = GridSearchCV(gbr, param_grid, refit = True, verbose = 3, n_jobs = -1) 
  
# fitting the model for grid search 
grid.fit(X_train, y_train)


# %%
# Best parameter after hyper parameter tuning 
print(grid.best_params_) 
  
# Moel Parameters 
print(grid.best_estimator_)


# %%
# Prediction using best parameters
grid_predictions = grid.predict(X_test) 
  
print("r2 score : ",r2_score(y_test,grid_predictions))
print("MAPE     : ",mean_absolute_percentage_error(y_test,grid_predictions))


# %%
import pickle


# %%
pickle.dump(gbr, open('optimalvehiclepricing.pkl','wb'))


# %%
model = pickle.load(open('optimalvehiclepricing.pkl','rb'))


# %%



