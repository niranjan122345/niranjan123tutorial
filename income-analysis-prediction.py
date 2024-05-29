import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, KFold
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_score, recall_score, confusion_matrix, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from pydantic_settings import BaseSettings

warnings.filterwarnings("ignore")
print("All dependencies are imported.")

# The rest of your code remains the same

# In[ ]:


train=pd.read_csv(r"C:\Users\user\Downloads\income prediction\train.csv")
print("Dataset for training:")
print(train)


# In[ ]:


print("Train DF Information:")
print(train.info())


# In[ ]:


print("Null counts:")
print(train.isnull().sum())


# In[ ]:


print("Train DF Description:")
print(train.describe())


# In[ ]:


print("Train DF Description:")
print(train.describe(include="object"))


# In[ ]:


report=ProfileReport(train)
print("Preliminary Analysis:")
print(report)


# In[ ]:


print("DF Columns:")
print(list(train.columns))


# In[ ]:


cat_cols=["workclass","education","marital-status","occupation","relationship","race","gender",
          "native-country","income_>50K"]
num_cols=[col for col in list(train.columns) if col not in cat_cols]
print("Categorical columns:")
print(cat_cols)
print("Numerical Columns")
print(num_cols)


# In[ ]:


print("Number of categorical columns:")
print(len(cat_cols))
print("Number of numerical columns:")
print(len(num_cols))


# In[ ]:


fig, axes = plt.subplots(5, 2, figsize=(20, 50))
for i, col in enumerate(cat_cols):
    r = i // 2  
    c = i % 2   
    sns.countplot(x=col, data=train, ax=axes[r, c])
    for container in axes[r, c].containers:
        axes[r, c].bar_label(container, label_type="edge")
    axes[r, c].set_xlabel(f"{col}")
    axes[r, c].set_ylabel("Frequency")
    axes[r, c].set_xticklabels(axes[r,c].get_xticklabels(),rotation="vertical")
    axes[r, c].set_title(f"{cat_cols[i]} Distribution")
if len(cat_cols) % 2 != 0:
    fig.delaxes(axes[-1, -1])
plt.subplots_adjust(hspace=1)
fig.text(0.5,0.9,"Categorical Columns Distribution (Bar Graph)", va="center", ha="center", fontsize=14)
plt.show()


# In[ ]:


fig, axes = plt.subplots(5, 2, figsize=(20, 40))
for i, col in enumerate(cat_cols):
    r = i // 2
    c = i % 2
    counts = train[col].value_counts()
    wedges, texts, autotexts = axes[r, c].pie(counts, labels=None, autopct='', startangle=90)
    axes[r, c].set_title(f"{col} Distribution")
    legend_labels = [f'{index}: {value:.1%}' for index, value in zip(counts.index, counts / counts.sum())]
    axes[r, c].legend(wedges, legend_labels, title=f'{col} Distribution', loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
if len(cat_cols) % 2 != 0:
    fig.delaxes(axes[-1, -1])
plt.subplots_adjust(hspace=2, wspace=1)
fig.text(0.5,0.95,"Few Categorical Columns Distribution (Pie Chart)",va="center", ha="center", fontsize=14)
plt.show()


# In[ ]:


fig, axes = plt.subplots(3, 2, figsize=(10, 20))
for i, col in enumerate(num_cols):
    r,c=i//2,i%2
    sns.histplot(x=col, data=train,ax=axes[r,c])
    axes[i//2,i%2].set_ylabel("Density")
    axes[i//2,i%2].set_title(f"{col} Density")
if len(num_cols) % 2 != 0:
    fig.delaxes(axes[-1, -1])
plt.subplots_adjust(hspace=0.5, wspace=0.5)
fig.text(0.5,0.95,"Numerical Columns Distribution",va="center", ha="center", fontsize=14)
plt.show()


# In[ ]:


train.dropna(inplace=True, axis=0)
print("Dropping the null values as it has low ratio:")
print(train.isnull().sum())


# In[ ]:


fig, axes = plt.subplots(4, 2, figsize=(20, 50))
for i, col in enumerate(cat_cols[:-1]):
    r = i // 2  
    c = i % 2   
    sns.countplot(x=col, data=train, hue=cat_cols[-1], ax=axes[r, c])
    for container in axes[r, c].containers:
        axes[r, c].bar_label(container, label_type="edge")
    axes[r, c].set_xlabel(f"{col}")
    axes[r, c].set_ylabel("Frequency")
    axes[r, c].set_xticklabels(axes[r,c].get_xticklabels(),rotation="vertical")
    axes[r, c].set_title(f"{cat_cols[i]} Distribution")
if len(cat_cols) % 2 != 0:
    fig.delaxes(axes[-1, -1])
plt.subplots_adjust(hspace=1)
fig.text(0.5,0.9,"Categorical Columns Distribution (Bar Graph)", va="center", ha="center", fontsize=14)
plt.show()


# In[ ]:


fig, axes = plt.subplots(3, 2, figsize=(10, 20))
for i, col in enumerate(num_cols):
    r,c=i//2,i%2
    sns.histplot(x=col, hue="income_>50K", data=train,ax=axes[r,c])
    axes[i//2,i%2].set_ylabel("Density")
    axes[i//2,i%2].set_title(f"{col} Density")
if len(num_cols) % 2 != 0:
    fig.delaxes(axes[-1, -1])
plt.subplots_adjust(hspace=0.5, wspace=0.5)
fig.text(0.5,0.95,"Numerical Columns Distribution W.R.T Income",va="center", ha="center", fontsize=14)
plt.show()


# In[ ]:


num_col = train.select_dtypes(include=['number']).columns
corr_mat = train[num_col].corr()
plt.figure(figsize = (18,18))
sns.heatmap(corr_mat, annot = True,cmap="coolwarm", fmt="0.2f")
plt.title("Correlation Matrix")
plt.show()


# In[ ]:


new_cat_cols = train.select_dtypes(include=['object']).columns
print("Categorical columns to encode:")
print(new_cat_cols)


# In[ ]:


train= pd.get_dummies(train, columns = new_cat_cols)
print("One Hot encoding categorical columns:")
print(train)
new_cols = train.columns
print("New columns:")
print(new_cols)


# In[ ]:


correlation_matrix = train.corr()
correlation_with_target = correlation_matrix['income_>50K']
pivot_table = correlation_with_target.drop('income_>50K').reset_index()
pivot_table_sorted = pivot_table.sort_values(by='income_>50K',ascending=False)
print("Correlation of each column with 'income_>50K' (Ascending Order):")
print(pivot_table_sorted)


# In[ ]:


x=train.drop("income_>50K", axis=1)
y=train["income_>50K"]
print("Input Variable:")
print(x)
print()
print("Target Variable:")
print(y)


# In[ ]:


print("Over-sampling to resolve dataset imbalance:")
ros = RandomOverSampler()
x_ros, y_ros = ros.fit_resample(x, y)
print("Input variables:")
print(x_ros)
print()
print("Target variable:")
print(y_ros)


# ## Model Training & Evaluation

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_ros,y_ros,test_size=0.25,random_state=42)
print("Training Input Variables:")
print(x_train)
print()
print("Training Output Variables:")
print(y_test)
print()
print("Testing Input Variables:")
print(x_test)
print()
print("Testing Output Variables:")
print(y_test)


# In[ ]:


print("Standard scaling for better accuracy:")
scaler = StandardScaler()
x_train_sc = scaler.fit_transform(x_train)
x_test_sc = scaler.transform(x_test)
print("Training input variables scaled:")
print(x_train_sc)
print()
print("Testing input variables scaled:")
print(x_test_sc)


# In[ ]:


models=[LogisticRegression(),KNeighborsClassifier(), SVC(), DecisionTreeClassifier(), 
        RandomForestClassifier(), ExtraTreesClassifier()]
accuracies=[]
print
for model in models:
    print("Model Used:")
    print(model)
    model.fit(x_train_sc, y_train)
    y_pred=model.predict(x_test_sc)
    acc=round(accuracy_score(y_test, y_pred),4)
    print(f"Accuracy Acquired: {acc}.")
    accuracies.append(acc)
    print()
max_acc=max(accuracies)
print(f"Best Accuracy Score Recorded: {max_acc}.")
max_idx=accuracies.index(max_acc)
best_model=models[max_idx]
print()
print(f"Best Model Perfomance: {best_model}.")


# In[ ]:


print("Training with best model:")
print(best_model)
best_model.fit(x_train_sc, y_train)
y_pred=best_model.predict(x_test_sc)
accuracy = round(accuracy_score(y_test, y_pred),4)
precision = round(precision_score(y_test, y_pred),4)
recall = round(recall_score(y_test, y_pred),4)
cm=confusion_matrix(y_test, y_pred)
print("Best Model Evaluations:")
print(f"Accuracy is: {accuracy}.")
print(f"Precision is: {precision}.")
print(f"Recall is: {recall}.")
print("Classification report:")
print(classification_report(y_test, y_pred))


# In[ ]:


plt.figure(figsize = (10,10))
sns.heatmap(cm, annot = True,cmap="coolwarm", fmt="0.2f")
plt.title("Confusion Matrix")
plt.show()


# ## Predictive Model

# In[ ]:


test=pd.read_csv(r"C:\Users\user\Downloads\income prediction\test.csv")
print("Dataframe for testing:")
print(test)


# In[ ]:


print("Test DF Information:")
print(test.info())


# In[ ]:


print("Test DF Description:")
print(test.describe())


# In[ ]:


test_dummy = pd.get_dummies(test, columns=new_cat_cols)
print("Encoding the categorical variables:")
print(test_dummy)


# In[ ]:


test_dummy_cols=list(test_dummy.columns)
new_cols=list(new_cols)
print("Number of Test columns:")
print(len(test_dummy_cols))
print("Number of Train columns:")
print(len(new_cols))
print("Columns in test but not in new_cols:")
print(set(test_dummy_cols) - set(new_cols))
print("Columns in new_cols but not in test:")
print(set(new_cols) - set(test_dummy_cols))


# In[ ]:


test_dummy = test_dummy.reindex(columns=new_cols, fill_value=0)
print("Number of columns in test columns after re-arrangement:")
print(len(list(test_dummy.columns)))


# In[ ]:


print("Dropping the output column that got added due to the rearrangement:")
test_dummy.drop("income_>50K",axis=1, inplace=True)
pred_income=best_model.predict(test_dummy)
print("Making predictions:")
print(pred_income)
print()
print("Making the predictions and concatinating to the test DF:")
test["income_>50K"]=pred_income
print(test)


# In[ ]:


print("Saving the test DF along with predictions into CSV file.")
test.to_csv(r"C:\Users\user\Downloads\income prediction\predictions.csv")

