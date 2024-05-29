# Description

Salary data is important for many businesses as it can play a significant role in distribution of their resources. For example, a high end product needs to market their goods only to people with high income. A charitable organization may be focused to invest in areas where people make less money. Likewise, it will also help businesses to determine prices of its products and forecast sales. Governments may also be interested to know the status-quo of their citizens to carry out effective economic and development plans. Since the project determines whether a person makes more than 50K or not, we are determining the dependent categorical variable. Here, we will first collect the relevant data set. Once we collect the data, we will prepare the data using pandas, numpy and visualize the data to determine the relationship between the variables. After studying the data set, we will preprocess the data using python where we will eliminate the outliers and manage missing values. Once we successfully preprocess the data, we will make use of python’s supervised learning packages and algorithms to build our model. Once we prepare the model, we will access the model based upon its accuracy, sensitivity, and specificity. If the model meets our requirements, we will deploy the model to make predictions on income level of individuals.

# About Dataset

The original dataset was obtained from the 1994 U.S. Census database. The census is performed every decade within the United States. It gathers information on the population living within the country. The information captured from the census consists of data on age, gender, country of origin, marital status, housing conditions, marriage, education, employment, etc. The sneakpeak of the dataset is as follows:
Train DF Information:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 43957 entries, 0 to 43956
Data columns (total 15 columns):
 #   Column           Non-Null Count  Dtype 
---  ------           --------------  ----- 
 0   age              43957 non-null  int64 
 1   workclass        41459 non-null  object
 2   fnlwgt           43957 non-null  int64 
 3   education        43957 non-null  object
 4   educational-num  43957 non-null  int64 
 5   marital-status   43957 non-null  object
 6   occupation       41451 non-null  object
 7   relationship     43957 non-null  object
 8   race             43957 non-null  object
 9   gender           43957 non-null  object
 10  capital-gain     43957 non-null  int64 
 11  capital-loss     43957 non-null  int64 
 12  hours-per-week   43957 non-null  int64 
 13  native-country   43194 non-null  object
 14  income_>50K      43957 non-null  int64 
dtypes: int64(7), object(8)
memory usage: 5.0+ MB
None
