## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
<img width="202" height="242" alt="image" src="https://github.com/user-attachments/assets/1fb69a0e-2254-4a5e-8389-54aae8f7e5fd" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="221" height="222" alt="image" src="https://github.com/user-attachments/assets/c4bfefe0-17c9-41cd-a030-5f7aa8296678" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="396" height="407" alt="image" src="https://github.com/user-attachments/assets/aa7cfe69-2552-4bfd-acac-13ad1e90c0d8" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="252" height="236" alt="image" src="https://github.com/user-attachments/assets/602c9bc4-64bd-4d7c-9a58-e6be6ffc3775" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="291" height="237" alt="image" src="https://github.com/user-attachments/assets/6b20c769-40a3-4d76-8b0f-739bcb92d9f7" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="1692" height="347" alt="image" src="https://github.com/user-attachments/assets/0a1bcbbb-eb7c-4f4a-816e-8763fc1e5dc3" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
fb=pd.concat([df,nd],axis=1)
dfb=df.copy()
dfb
```
<img width="340" height="238" alt="image" src="https://github.com/user-attachments/assets/8afc9304-a8d1-4611-a892-0c662a2389b7" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
<img width="367" height="228" alt="image" src="https://github.com/user-attachments/assets/45ad0f4c-e1fa-4de1-9d6e-820be6bced32" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
<img width="508" height="271" alt="image" src="https://github.com/user-attachments/assets/b478c4fb-db5d-4b99-b468-70df997caf7e" />

```
df.skew()
```
<img width="198" height="88" alt="image" src="https://github.com/user-attachments/assets/1debeccc-7698-4588-930c-37ea4d37c90c" />

```
np.log(df["Highly Positive Skew"])
```
<img width="321" height="167" alt="image" src="https://github.com/user-attachments/assets/912bf6c7-efad-4273-8808-bfb98b561b3a" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="350" height="177" alt="image" src="https://github.com/user-attachments/assets/b2c29c8c-c41d-463f-a3f7-008cc8f50219" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="362" height="167" alt="image" src="https://github.com/user-attachments/assets/87cecc70-6974-4f97-8e8f-81fdf253100c" />

```
np.square(df["Highly Positive Skew"])
```
<img width="320" height="155" alt="image" src="https://github.com/user-attachments/assets/3d572909-dc3d-4cde-8b78-0cdb8db26365" />

```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
```
<img width="721" height="290" alt="image" src="https://github.com/user-attachments/assets/b42264f5-7a6a-40ba-9683-300204becbe3" />

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
<img width="263" height="105" alt="image" src="https://github.com/user-attachments/assets/681bfb35-fec2-42ec-98ca-6c3b47c4d27a" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

<img width="275" height="125" alt="image" src="https://github.com/user-attachments/assets/959e2f84-90cf-4e73-a04c-bea96bd0cee7" />

```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()
```
<img width="461" height="298" alt="image" src="https://github.com/user-attachments/assets/0bbd58ed-e189-41c6-a3ae-b12832314e9c" />

```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()
```

<img width="461" height="301" alt="image" src="https://github.com/user-attachments/assets/afe85226-a044-47f8-8dd7-8d35c794b6ea" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
```
<img width="442" height="437" alt="image" src="https://github.com/user-attachments/assets/90574f42-6e90-4edd-a35c-55423695cf76" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="422" height="262" alt="image" src="https://github.com/user-attachments/assets/53375245-6e43-4614-88df-1becaeb85200" />

## Result:
Hence performing Feature Encoding and Transformation process is Successful.
