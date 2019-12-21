
# Northwind Marketing Tactics and Strategies

# Applying the 4Ps of marketing 

* R. Mih 
* Cohort ft-ds-100719
* Instructor:  James Irving

* https://www.singlegrain.com/marketing/the-4-ps-of-marketing-are-they-still-relevant-today-price-product-promotion-place/


## Objective

### Use statistical data to determine the best way to position discounts to win business
### Following the "4Ps"  (Product, Price, Position, Placement) of Marketing


<img src='The 4 Ps of marketing.png' width=40%/>

##  Contents

### ETL and ERD

### Hypotheses

#### 1. Does the discount have a statistically significant effect on the order quantity?
        If yes, at what level(s) of discount

#### 2. Does the discount have a statistically significant effect on the customer product spend (product sales

#### 3. What's the best approach for pricing,  in a competitive environment

#### 4. Does a lower MSRP have a statistically significant effect on the order quantity and product spend (product sales)?

#### 5. Does the Net Discount have a statistically significant effect on order quantity and product spend (product sales)?


```python
#!pip install -U fsds_100719
from fsds_100719.imports import *
```

    fsds_1007219  v0.5.11 loaded.  Read the docs: https://fsds.readthedocs.io/en/latest/ 
    


<style  type="text/css" >
</style>  
<table id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8" ><caption>Loaded Packages and Handles</caption> 
<thead>    <tr> 
        <th class="col_heading level0 col0" >Handle</th> 
        <th class="col_heading level0 col1" >Package</th> 
        <th class="col_heading level0 col2" >Description</th> 
    </tr></thead> 
<tbody>    <tr> 
        <td id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8row0_col0" class="data row0 col0" >dp</td> 
        <td id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8row0_col1" class="data row0 col1" >IPython.display</td> 
        <td id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8row0_col2" class="data row0 col2" >Display modules with helpful display and clearing commands.</td> 
    </tr>    <tr> 
        <td id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8row1_col0" class="data row1 col0" >fs</td> 
        <td id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8row1_col1" class="data row1 col1" >fsds_100719</td> 
        <td id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8row1_col2" class="data row1 col2" >Custom data science bootcamp student package</td> 
    </tr>    <tr> 
        <td id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8row2_col0" class="data row2 col0" >mpl</td> 
        <td id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8row2_col1" class="data row2 col1" >matplotlib</td> 
        <td id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8row2_col2" class="data row2 col2" >Matplotlib's base OOP module with formatting artists</td> 
    </tr>    <tr> 
        <td id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8row3_col0" class="data row3 col0" >plt</td> 
        <td id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8row3_col1" class="data row3 col1" >matplotlib.pyplot</td> 
        <td id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8row3_col2" class="data row3 col2" >Matplotlib's matlab-like plotting module</td> 
    </tr>    <tr> 
        <td id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8row4_col0" class="data row4 col0" >np</td> 
        <td id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8row4_col1" class="data row4 col1" >numpy</td> 
        <td id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8row4_col2" class="data row4 col2" >scientific computing with Python</td> 
    </tr>    <tr> 
        <td id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8row5_col0" class="data row5 col0" >pd</td> 
        <td id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8row5_col1" class="data row5 col1" >pandas</td> 
        <td id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8row5_col2" class="data row5 col2" >High performance data structures and tools</td> 
    </tr>    <tr> 
        <td id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8row6_col0" class="data row6 col0" >sns</td> 
        <td id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8row6_col1" class="data row6 col1" >seaborn</td> 
        <td id="T_1dc59980_23b8_11ea_aee8_5800e3d0a9a8row6_col2" class="data row6 col2" >High-level data visualization library based on matplotlib</td> 
    </tr></tbody> 
</table> 



```python
import pandas as pd
import sqlite3
import numpy as np
from scipy import stats
import statsmodels.api as sms
import statsmodels.formula.api as smf
import scipy.stats as stat
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels as sms

sns.set_style('whitegrid')
pd.set_eng_float_format(accuracy=3, use_eng_prefix=True)

%config InlineBackend.figure_format = 'retina'
%matplotlib inline

# Set pandas options
pd.set_option('display.precision',3)
pd.set_option('display.max_columns',0)

# Filter warnings
import warnings
warnings.filterwarnings('ignore')

conn = sqlite3.connect('Northwind_small.sqlite')
cur = conn.cursor()

# Set Constants

alpha = 0.05
```

<div>
<img src="Northwind_ERD_updated.png" width="1000"/>
</div


```python
# Local Functions to be used

def load_data(select_text, disp_rows =5):
    try:
        cur.execute(select_text)
        new_df = pd.DataFrame(cur.fetchall())
        new_df.columns = [x[0] for x in cur.description]
        print("Size of the dataframe = ", new_df.shape)
        print("Number of null data")
        print(new_df.isnull().sum())
        print("Number of duplicated data = ", new_df.duplicated().sum())
        display(new_df.head(disp_rows))
        return new_df
    except:
        print("Table not loaded")
    

```

# ETL Methodology 

#### Inspect each table and confirm the ERD is correct wrt the column names
#### Inspect the data in each table
    A.  Review the size of the dataframe
    B.  Inspect for null and duplicated data
#### For larger tables, inspect the distribution of data in each column using value_counts()
#### Summarize the data

# Product Table
#### There are 77 products


```python
# Product Table
select_text = 'SELECT * FROM Product;'
product_df = load_data(select_text)
```

    Size of the dataframe =  (77, 10)
    Number of null data
    Id                 0
    ProductName        0
    SupplierId         0
    CategoryId         0
    QuantityPerUnit    0
    UnitPrice          0
    UnitsInStock       0
    UnitsOnOrder       0
    ReorderLevel       0
    Discontinued       0
    dtype: int64
    Number of duplicated data =  0
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>ProductName</th>
      <th>SupplierId</th>
      <th>CategoryId</th>
      <th>QuantityPerUnit</th>
      <th>UnitPrice</th>
      <th>UnitsInStock</th>
      <th>UnitsOnOrder</th>
      <th>ReorderLevel</th>
      <th>Discontinued</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Chai</td>
      <td>1</td>
      <td>1</td>
      <td>10 boxes x 20 bags</td>
      <td>18.000</td>
      <td>39</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Chang</td>
      <td>1</td>
      <td>1</td>
      <td>24 - 12 oz bottles</td>
      <td>19.000</td>
      <td>17</td>
      <td>40</td>
      <td>25</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Aniseed Syrup</td>
      <td>1</td>
      <td>2</td>
      <td>12 - 550 ml bottles</td>
      <td>10.000</td>
      <td>13</td>
      <td>70</td>
      <td>25</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Chef Anton's Cajun Seasoning</td>
      <td>2</td>
      <td>2</td>
      <td>48 - 6 oz jars</td>
      <td>22.000</td>
      <td>53</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Chef Anton's Gumbo Mix</td>
      <td>2</td>
      <td>2</td>
      <td>36 boxes</td>
      <td>21.350</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
ax = product_df.hist(figsize=(9,9))
```


![png](output_11_0.png)



```python
## Our goal is a marketing report rather than an operations report,  so removing columns regarding stock and reorders
product_df.drop(columns = ['UnitsInStock', 'UnitsOnOrder', 'ReorderLevel', 'Discontinued' ], inplace=True)
```

# Supplier Table
#### There are 29 suppliers


```python
select_text = 'SELECT * FROM Supplier;'
supplier_df = load_data(select_text)
```

    Size of the dataframe =  (29, 12)
    Number of null data
    Id               0
    CompanyName      0
    ContactName      0
    ContactTitle     0
    Address          0
    City             0
    Region           0
    PostalCode       0
    Country          0
    Phone            0
    Fax             16
    HomePage        24
    dtype: int64
    Number of duplicated data =  0
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>CompanyName</th>
      <th>ContactName</th>
      <th>ContactTitle</th>
      <th>Address</th>
      <th>City</th>
      <th>Region</th>
      <th>PostalCode</th>
      <th>Country</th>
      <th>Phone</th>
      <th>Fax</th>
      <th>HomePage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Exotic Liquids</td>
      <td>Charlotte Cooper</td>
      <td>Purchasing Manager</td>
      <td>49 Gilbert St.</td>
      <td>London</td>
      <td>British Isles</td>
      <td>EC1 4SD</td>
      <td>UK</td>
      <td>(171) 555-2222</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>New Orleans Cajun Delights</td>
      <td>Shelley Burke</td>
      <td>Order Administrator</td>
      <td>P.O. Box 78934</td>
      <td>New Orleans</td>
      <td>North America</td>
      <td>70117</td>
      <td>USA</td>
      <td>(100) 555-4822</td>
      <td>None</td>
      <td>#CAJUN.HTM#</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grandma Kelly's Homestead</td>
      <td>Regina Murphy</td>
      <td>Sales Representative</td>
      <td>707 Oxford Rd.</td>
      <td>Ann Arbor</td>
      <td>North America</td>
      <td>48104</td>
      <td>USA</td>
      <td>(313) 555-5735</td>
      <td>(313) 555-3349</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Tokyo Traders</td>
      <td>Yoshi Nagase</td>
      <td>Marketing Manager</td>
      <td>9-8 Sekimai Musashino-shi</td>
      <td>Tokyo</td>
      <td>Eastern Asia</td>
      <td>100</td>
      <td>Japan</td>
      <td>(03) 3555-5011</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Cooperativa de Quesos 'Las Cabras'</td>
      <td>Antonio del Valle Saavedra</td>
      <td>Export Administrator</td>
      <td>Calle del Rosal 4</td>
      <td>Oviedo</td>
      <td>Southern Europe</td>
      <td>33007</td>
      <td>Spain</td>
      <td>(98) 598 76 54</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>


# Region Table
#### 4 Regions (East, West, North, South)


```python
select_data = 'SELECT * FROM Region;'
region_df = load_data(select_data)
```

    Size of the dataframe =  (4, 2)
    Number of null data
    Id                   0
    RegionDescription    0
    dtype: int64
    Number of duplicated data =  0
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>RegionDescription</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Eastern</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Western</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Northern</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Southern</td>
    </tr>
  </tbody>
</table>
</div>


# Category Table

##### 8 Categories of edible goods



```python
select_data = 'SELECT * FROM Category;'
category_df = load_data(select_data)
```

    Size of the dataframe =  (8, 3)
    Number of null data
    Id              0
    CategoryName    0
    Description     0
    dtype: int64
    Number of duplicated data =  0
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>CategoryName</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Beverages</td>
      <td>Soft drinks, coffees, teas, beers, and ales</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Condiments</td>
      <td>Sweet and savory sauces, relishes, spreads, an...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Confections</td>
      <td>Desserts, candies, and sweet breads</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Dairy Products</td>
      <td>Cheeses</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Grains/Cereals</td>
      <td>Breads, crackers, pasta, and cereal</td>
    </tr>
  </tbody>
</table>
</div>


# Order Detail Table
#####  One of the largest tables

#####  2155 Products Sold



```python
select_data = 'SELECT * FROM OrderDetail;'
orderdetail_df = load_data(select_data)
```

    Size of the dataframe =  (2155, 6)
    Number of null data
    Id           0
    OrderId      0
    ProductId    0
    UnitPrice    0
    Quantity     0
    Discount     0
    dtype: int64
    Number of duplicated data =  0
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>OrderId</th>
      <th>ProductId</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10248/11</td>
      <td>10248</td>
      <td>11</td>
      <td>14.000</td>
      <td>12</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10248/42</td>
      <td>10248</td>
      <td>42</td>
      <td>9.800</td>
      <td>10</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10248/72</td>
      <td>10248</td>
      <td>72</td>
      <td>34.800</td>
      <td>5</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10249/14</td>
      <td>10249</td>
      <td>14</td>
      <td>18.600</td>
      <td>9</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10249/51</td>
      <td>10249</td>
      <td>51</td>
      <td>42.400</td>
      <td>40</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>


#### Value Counts for Order Detail

#### Display distribution of prices and discounts


```python
for col in orderdetail_df.columns:
    print(col, '\n', orderdetail_df[col].value_counts().head(2200), '\n')
    #orderdetail_df[col].value_counts().plot(kind='bar', figsize=(3,3));
```

    Id 
     10782/31    1
    11031/71    1
    10580/14    1
    10850/25    1
    10726/11    1
    10879/65    1
    10609/21    1
    10718/36    1
    10634/75    1
    10699/47    1
    11053/18    1
    10656/47    1
    10740/28    1
    10956/51    1
    10644/46    1
    10287/46    1
    10758/70    1
    10978/8     1
    10323/25    1
    10887/25    1
    10580/65    1
    10332/47    1
    11077/77    1
    10734/76    1
    10829/2     1
    10786/30    1
    10395/53    1
    10396/23    1
    10428/46    1
    10358/24    1
               ..
    10547/32    1
    11043/11    1
    10582/76    1
    10748/56    1
    10549/31    1
    10512/24    1
    10303/65    1
    10363/76    1
    10775/10    1
    10351/65    1
    10761/25    1
    10516/41    1
    10488/59    1
    10362/51    1
    10452/28    1
    11039/49    1
    10563/36    1
    10913/58    1
    10591/7     1
    10274/71    1
    10290/29    1
    10676/10    1
    10750/45    1
    10786/75    1
    10424/35    1
    10588/18    1
    10390/31    1
    10575/63    1
    10258/2     1
    10793/52    1
    Name: Id, Length: 2155, dtype: int64 
    
    OrderId 
     11077    25
    10657     6
    10979     6
    10847     6
    10861     5
    10558     5
    11031     5
    11021     5
    10393     5
    10612     5
    10962     5
    10623     5
    10670     5
    10893     5
    10465     5
    10337     5
    10698     5
    10845     5
    10714     5
    10515     5
    10537     5
    10553     5
    10555     5
    10691     5
    10607     5
    10294     5
    10836     5
    10382     5
    10514     5
    10360     5
             ..
    10433     1
    10969     1
    11061     1
    10427     1
    10674     1
    10292     1
    10883     1
    10881     1
    10782     1
    10778     1
    10899     1
    10942     1
    10457     1
    10683     1
    10914     1
    10660     1
    11067     1
    11069     1
    10669     1
    10662     1
    11074     1
    10920     1
    10907     1
    10905     1
    10568     1
    10679     1
    10934     1
    10266     1
    10936     1
    10759     1
    Name: OrderId, Length: 830, dtype: int64 
    
    ProductId 
     59    54
    31    51
    24    51
    60    51
    56    50
    62    48
    41    47
    75    46
    2     44
    16    43
    71    42
    40    41
    13    40
    70    39
    76    39
    51    39
    21    39
    72    38
    11    38
    77    38
    1     38
    19    37
    17    37
    54    36
    35    36
    68    34
    55    33
    28    33
    10    33
    29    32
          ..
    38    24
    57    23
    14    22
    49    21
    47    21
    23    20
    4     20
    34    19
    58    18
    25    18
    63    17
    20    16
    32    15
    12    14
    73    14
    45    14
    22    14
    8     13
    74    13
    3     12
    6     12
    67    10
    50    10
    5     10
    27     9
    66     8
    48     6
    15     6
    37     6
    9      5
    Name: ProductId, Length: 77, dtype: int64 
    
    UnitPrice 
     18.000     102
    10.000      71
    14.000      56
    12.500      55
    19.000      53
    38.000      45
    14.400      41
    15.000      39
    21.000      38
    34.000      37
    9.650       36
    4.500       36
    55.000      35
    7.750       35
    49.300      34
    6.000       32
    12.000      31
    53.000      30
    17.450      30
    9.500       29
    13.000      29
    18.400      28
    31.000      26
    24.000      26
    30.000      25
    39.000      25
    21.500      25
    16.800      25
    9.200       24
    7.000       24
              ... 
    42.400       9
    35.100       9
    18.600       9
    36.800       9
    16.000       8
    17.000       8
    4.800        8
    15.600       8
    210.800      8
    24.800       7
    7.200        7
    50.000       6
    7.600        6
    21.350       6
    17.600       6
    16.250       6
    5.600        5
    64.800       5
    97.000       4
    13.600       4
    12.750       4
    10.600       4
    22.800       3
    26.000       3
    20.800       3
    25.600       3
    12.400       2
    10.200       2
    77.600       1
    9.800        1
    Name: UnitPrice, Length: 116, dtype: int64 
    
    Quantity 
     20     252
    30     194
    10     181
    15     169
    40     113
    12      92
    6       87
    25      80
    50      75
    35      71
    5       67
    24      58
    60      58
    4       55
    2       52
    21      50
    8       50
    18      47
    3       46
    14      36
    16      35
    9       30
    28      29
    70      28
    7       23
    36      21
    42      19
    1       17
    80      16
    45      14
    100     10
    55       9
    120      8
    65       8
    49       7
    48       5
    32       5
    44       4
    90       4
    13       3
    77       3
    110      3
    56       3
    84       2
    11       2
    27       2
    33       2
    39       2
    130      2
    63       1
    66       1
    22       1
    52       1
    54       1
    91       1
    Name: Quantity, dtype: int64 
    
    Discount 
     0.000       1317
    50.000m      185
    100.000m     173
    200.000m     161
    150.000m     157
    250.000m     154
    30.000m        3
    20.000m        2
    10.000m        1
    40.000m        1
    60.000m        1
    Name: Discount, dtype: int64 
    
    

#### Feature engineering.  

#### We want to manipulate the data to compare populations with and without discounts shown in the OrderDetail
#### Add a categorical column  called "Discounts" for discounts vs no discounts
#### Add a Final Price column
#### Add a total product spend column
#### Plot a histogram of the data for visual inspection
    * >800 products were discounted, with the ranges of discounts from 5%-25%
    * Bins for significant discounts looks like 5,10,15,20,25


```python
orderdetail_df['Discounts'] = np.where(orderdetail_df['Discount'] == 0,0,1)
orderdetail_df['Final Price'] = orderdetail_df['UnitPrice'] * (1 - orderdetail_df['Discount'])
orderdetail_df['Total Product Spend ($)'] = orderdetail_df['Final Price'] * orderdetail_df['Quantity']
orderdetail_df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>OrderId</th>
      <th>ProductId</th>
      <th>UnitPrice</th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>Discounts</th>
      <th>Final Price</th>
      <th>Total Product Spend ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10248/11</td>
      <td>10248</td>
      <td>11</td>
      <td>14.000</td>
      <td>12</td>
      <td>0.000</td>
      <td>0</td>
      <td>14.000</td>
      <td>168.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10248/42</td>
      <td>10248</td>
      <td>42</td>
      <td>9.800</td>
      <td>10</td>
      <td>0.000</td>
      <td>0</td>
      <td>9.800</td>
      <td>98.000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10248/72</td>
      <td>10248</td>
      <td>72</td>
      <td>34.800</td>
      <td>5</td>
      <td>0.000</td>
      <td>0</td>
      <td>34.800</td>
      <td>174.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10249/14</td>
      <td>10249</td>
      <td>14</td>
      <td>18.600</td>
      <td>9</td>
      <td>0.000</td>
      <td>0</td>
      <td>18.600</td>
      <td>167.400</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10249/51</td>
      <td>10249</td>
      <td>51</td>
      <td>42.400</td>
      <td>40</td>
      <td>0.000</td>
      <td>0</td>
      <td>42.400</td>
      <td>1.696k</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10250/41</td>
      <td>10250</td>
      <td>41</td>
      <td>7.700</td>
      <td>10</td>
      <td>0.000</td>
      <td>0</td>
      <td>7.700</td>
      <td>77.000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10250/51</td>
      <td>10250</td>
      <td>51</td>
      <td>42.400</td>
      <td>35</td>
      <td>150.000m</td>
      <td>1</td>
      <td>36.040</td>
      <td>1.261k</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10250/65</td>
      <td>10250</td>
      <td>65</td>
      <td>16.800</td>
      <td>15</td>
      <td>150.000m</td>
      <td>1</td>
      <td>14.280</td>
      <td>214.200</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10251/22</td>
      <td>10251</td>
      <td>22</td>
      <td>16.800</td>
      <td>6</td>
      <td>50.000m</td>
      <td>1</td>
      <td>15.960</td>
      <td>95.760</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10251/57</td>
      <td>10251</td>
      <td>57</td>
      <td>15.600</td>
      <td>15</td>
      <td>50.000m</td>
      <td>1</td>
      <td>14.820</td>
      <td>222.300</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10251/65</td>
      <td>10251</td>
      <td>65</td>
      <td>16.800</td>
      <td>20</td>
      <td>0.000</td>
      <td>0</td>
      <td>16.800</td>
      <td>336.000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>10252/20</td>
      <td>10252</td>
      <td>20</td>
      <td>64.800</td>
      <td>40</td>
      <td>50.000m</td>
      <td>1</td>
      <td>61.560</td>
      <td>2.462k</td>
    </tr>
    <tr>
      <th>12</th>
      <td>10252/33</td>
      <td>10252</td>
      <td>33</td>
      <td>2.000</td>
      <td>25</td>
      <td>50.000m</td>
      <td>1</td>
      <td>1.900</td>
      <td>47.500</td>
    </tr>
    <tr>
      <th>13</th>
      <td>10252/60</td>
      <td>10252</td>
      <td>60</td>
      <td>27.200</td>
      <td>40</td>
      <td>0.000</td>
      <td>0</td>
      <td>27.200</td>
      <td>1.088k</td>
    </tr>
    <tr>
      <th>14</th>
      <td>10253/31</td>
      <td>10253</td>
      <td>31</td>
      <td>10.000</td>
      <td>20</td>
      <td>0.000</td>
      <td>0</td>
      <td>10.000</td>
      <td>200.000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>10253/39</td>
      <td>10253</td>
      <td>39</td>
      <td>14.400</td>
      <td>42</td>
      <td>0.000</td>
      <td>0</td>
      <td>14.400</td>
      <td>604.800</td>
    </tr>
    <tr>
      <th>16</th>
      <td>10253/49</td>
      <td>10253</td>
      <td>49</td>
      <td>16.000</td>
      <td>40</td>
      <td>0.000</td>
      <td>0</td>
      <td>16.000</td>
      <td>640.000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>10254/24</td>
      <td>10254</td>
      <td>24</td>
      <td>3.600</td>
      <td>15</td>
      <td>150.000m</td>
      <td>1</td>
      <td>3.060</td>
      <td>45.900</td>
    </tr>
    <tr>
      <th>18</th>
      <td>10254/55</td>
      <td>10254</td>
      <td>55</td>
      <td>19.200</td>
      <td>21</td>
      <td>150.000m</td>
      <td>1</td>
      <td>16.320</td>
      <td>342.720</td>
    </tr>
    <tr>
      <th>19</th>
      <td>10254/74</td>
      <td>10254</td>
      <td>74</td>
      <td>8.000</td>
      <td>21</td>
      <td>0.000</td>
      <td>0</td>
      <td>8.000</td>
      <td>168.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = orderdetail_df.hist(figsize=(6,9))
```


![png](output_25_0.png)


# Employees Table
####  9 Employees



```python
select_data = 'SELECT * FROM Employee;'
Employee_df = load_data(select_data)
```

    Size of the dataframe =  (9, 18)
    Number of null data
    Id                 0
    LastName           0
    FirstName          0
    Title              0
    TitleOfCourtesy    0
    BirthDate          0
    HireDate           0
    Address            0
    City               0
    Region             0
    PostalCode         0
    Country            0
    HomePhone          0
    Extension          0
    Photo              9
    Notes              0
    ReportsTo          1
    PhotoPath          0
    dtype: int64
    Number of duplicated data =  0
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>LastName</th>
      <th>FirstName</th>
      <th>Title</th>
      <th>TitleOfCourtesy</th>
      <th>BirthDate</th>
      <th>HireDate</th>
      <th>Address</th>
      <th>City</th>
      <th>Region</th>
      <th>PostalCode</th>
      <th>Country</th>
      <th>HomePhone</th>
      <th>Extension</th>
      <th>Photo</th>
      <th>Notes</th>
      <th>ReportsTo</th>
      <th>PhotoPath</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Davolio</td>
      <td>Nancy</td>
      <td>Sales Representative</td>
      <td>Ms.</td>
      <td>1980-12-08</td>
      <td>2024-05-01</td>
      <td>507 - 20th Ave. E. Apt. 2A</td>
      <td>Seattle</td>
      <td>North America</td>
      <td>98122</td>
      <td>USA</td>
      <td>(206) 555-9857</td>
      <td>5467</td>
      <td>None</td>
      <td>Education includes a BA in psychology from Col...</td>
      <td>2.000</td>
      <td>http://accweb/emmployees/davolio.bmp</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Fuller</td>
      <td>Andrew</td>
      <td>Vice President, Sales</td>
      <td>Dr.</td>
      <td>1984-02-19</td>
      <td>2024-08-14</td>
      <td>908 W. Capital Way</td>
      <td>Tacoma</td>
      <td>North America</td>
      <td>98401</td>
      <td>USA</td>
      <td>(206) 555-9482</td>
      <td>3457</td>
      <td>None</td>
      <td>Andrew received his BTS commercial in 1974 and...</td>
      <td>NaN</td>
      <td>http://accweb/emmployees/fuller.bmp</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Leverling</td>
      <td>Janet</td>
      <td>Sales Representative</td>
      <td>Ms.</td>
      <td>1995-08-30</td>
      <td>2024-04-01</td>
      <td>722 Moss Bay Blvd.</td>
      <td>Kirkland</td>
      <td>North America</td>
      <td>98033</td>
      <td>USA</td>
      <td>(206) 555-3412</td>
      <td>3355</td>
      <td>None</td>
      <td>Janet has a BS degree in chemistry from Boston...</td>
      <td>2.000</td>
      <td>http://accweb/emmployees/leverling.bmp</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Peacock</td>
      <td>Margaret</td>
      <td>Sales Representative</td>
      <td>Mrs.</td>
      <td>1969-09-19</td>
      <td>2025-05-03</td>
      <td>4110 Old Redmond Rd.</td>
      <td>Redmond</td>
      <td>North America</td>
      <td>98052</td>
      <td>USA</td>
      <td>(206) 555-8122</td>
      <td>5176</td>
      <td>None</td>
      <td>Margaret holds a BA in English literature from...</td>
      <td>2.000</td>
      <td>http://accweb/emmployees/peacock.bmp</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Buchanan</td>
      <td>Steven</td>
      <td>Sales Manager</td>
      <td>Mr.</td>
      <td>1987-03-04</td>
      <td>2025-10-17</td>
      <td>14 Garrett Hill</td>
      <td>London</td>
      <td>British Isles</td>
      <td>SW1 8JR</td>
      <td>UK</td>
      <td>(71) 555-4848</td>
      <td>3453</td>
      <td>None</td>
      <td>Steven Buchanan graduated from St. Andrews Uni...</td>
      <td>2.000</td>
      <td>http://accweb/emmployees/buchanan.bmp</td>
    </tr>
  </tbody>
</table>
</div>


# Employee Territory Table


```python
select_data = 'SELECT * FROM EmployeeTerritory;'
employee_territory_df = load_data(select_data)
```

    Size of the dataframe =  (49, 3)
    Number of null data
    Id             0
    EmployeeId     0
    TerritoryId    0
    dtype: int64
    Number of duplicated data =  0
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>EmployeeId</th>
      <th>TerritoryId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/06897</td>
      <td>1</td>
      <td>06897</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/19713</td>
      <td>1</td>
      <td>19713</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2/01581</td>
      <td>2</td>
      <td>01581</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2/01730</td>
      <td>2</td>
      <td>01730</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2/01833</td>
      <td>2</td>
      <td>01833</td>
    </tr>
  </tbody>
</table>
</div>


# Territory Table


```python
select_data = 'SELECT * FROM Territory;'
territory_df = load_data(select_data)

```

    Size of the dataframe =  (53, 3)
    Number of null data
    Id                      0
    TerritoryDescription    0
    RegionId                0
    dtype: int64
    Number of duplicated data =  0
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>TerritoryDescription</th>
      <th>RegionId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01581</td>
      <td>Westboro</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01730</td>
      <td>Bedford</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01833</td>
      <td>Georgetow</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>02116</td>
      <td>Boston</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>02139</td>
      <td>Cambridge</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


# Customer Table
#### 91 unique customers


```python
select_data = 'SELECT * FROM Customer;'
customer_df = load_data(select_data)
```

    Size of the dataframe =  (91, 11)
    Number of null data
    Id               0
    CompanyName      0
    ContactName      0
    ContactTitle     0
    Address          0
    City             0
    Region           0
    PostalCode       1
    Country          0
    Phone            0
    Fax             22
    dtype: int64
    Number of duplicated data =  0
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>CompanyName</th>
      <th>ContactName</th>
      <th>ContactTitle</th>
      <th>Address</th>
      <th>City</th>
      <th>Region</th>
      <th>PostalCode</th>
      <th>Country</th>
      <th>Phone</th>
      <th>Fax</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ALFKI</td>
      <td>Alfreds Futterkiste</td>
      <td>Maria Anders</td>
      <td>Sales Representative</td>
      <td>Obere Str. 57</td>
      <td>Berlin</td>
      <td>Western Europe</td>
      <td>12209</td>
      <td>Germany</td>
      <td>030-0074321</td>
      <td>030-0076545</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ANATR</td>
      <td>Ana Trujillo Emparedados y helados</td>
      <td>Ana Trujillo</td>
      <td>Owner</td>
      <td>Avda. de la Constitución 2222</td>
      <td>México D.F.</td>
      <td>Central America</td>
      <td>05021</td>
      <td>Mexico</td>
      <td>(5) 555-4729</td>
      <td>(5) 555-3745</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ANTON</td>
      <td>Antonio Moreno Taquería</td>
      <td>Antonio Moreno</td>
      <td>Owner</td>
      <td>Mataderos  2312</td>
      <td>México D.F.</td>
      <td>Central America</td>
      <td>05023</td>
      <td>Mexico</td>
      <td>(5) 555-3932</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AROUT</td>
      <td>Around the Horn</td>
      <td>Thomas Hardy</td>
      <td>Sales Representative</td>
      <td>120 Hanover Sq.</td>
      <td>London</td>
      <td>British Isles</td>
      <td>WA1 1DP</td>
      <td>UK</td>
      <td>(171) 555-7788</td>
      <td>(171) 555-6750</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BERGS</td>
      <td>Berglunds snabbköp</td>
      <td>Christina Berglund</td>
      <td>Order Administrator</td>
      <td>Berguvsvägen  8</td>
      <td>Luleå</td>
      <td>Northern Europe</td>
      <td>S-958 22</td>
      <td>Sweden</td>
      <td>0921-12 34 65</td>
      <td>0921-12 34 67</td>
    </tr>
  </tbody>
</table>
</div>


# Shipper Table
#### 3 Unique shippers


```python
select_data = 'SELECT * FROM Shipper;'
shipper_df = load_data(select_data)


```

    Size of the dataframe =  (3, 3)
    Number of null data
    Id             0
    CompanyName    0
    Phone          0
    dtype: int64
    Number of duplicated data =  0
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>CompanyName</th>
      <th>Phone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Speedy Express</td>
      <td>(503) 555-9831</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>United Package</td>
      <td>(503) 555-3199</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Federal Shipping</td>
      <td>(503) 555-9931</td>
    </tr>
  </tbody>
</table>
</div>


# Order Table
#### 830 Customers


```python
select_data = 'SELECT * FROM "Order" '
order_df = load_data(select_data)
```

    Size of the dataframe =  (830, 14)
    Number of null data
    Id                 0
    CustomerId         0
    EmployeeId         0
    OrderDate          0
    RequiredDate       0
    ShippedDate       21
    ShipVia            0
    Freight            0
    ShipName           0
    ShipAddress        0
    ShipCity           0
    ShipRegion         0
    ShipPostalCode    19
    ShipCountry        0
    dtype: int64
    Number of duplicated data =  0
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>CustomerId</th>
      <th>EmployeeId</th>
      <th>OrderDate</th>
      <th>RequiredDate</th>
      <th>ShippedDate</th>
      <th>ShipVia</th>
      <th>Freight</th>
      <th>ShipName</th>
      <th>ShipAddress</th>
      <th>ShipCity</th>
      <th>ShipRegion</th>
      <th>ShipPostalCode</th>
      <th>ShipCountry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10248</td>
      <td>VINET</td>
      <td>5</td>
      <td>2012-07-04</td>
      <td>2012-08-01</td>
      <td>2012-07-16</td>
      <td>3</td>
      <td>32.380</td>
      <td>Vins et alcools Chevalier</td>
      <td>59 rue de l'Abbaye</td>
      <td>Reims</td>
      <td>Western Europe</td>
      <td>51100</td>
      <td>France</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10249</td>
      <td>TOMSP</td>
      <td>6</td>
      <td>2012-07-05</td>
      <td>2012-08-16</td>
      <td>2012-07-10</td>
      <td>1</td>
      <td>11.610</td>
      <td>Toms Spezialitäten</td>
      <td>Luisenstr. 48</td>
      <td>Münster</td>
      <td>Western Europe</td>
      <td>44087</td>
      <td>Germany</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10250</td>
      <td>HANAR</td>
      <td>4</td>
      <td>2012-07-08</td>
      <td>2012-08-05</td>
      <td>2012-07-12</td>
      <td>2</td>
      <td>65.830</td>
      <td>Hanari Carnes</td>
      <td>Rua do Paço, 67</td>
      <td>Rio de Janeiro</td>
      <td>South America</td>
      <td>05454-876</td>
      <td>Brazil</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10251</td>
      <td>VICTE</td>
      <td>3</td>
      <td>2012-07-08</td>
      <td>2012-08-05</td>
      <td>2012-07-15</td>
      <td>1</td>
      <td>41.340</td>
      <td>Victuailles en stock</td>
      <td>2, rue du Commerce</td>
      <td>Lyon</td>
      <td>Western Europe</td>
      <td>69004</td>
      <td>France</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10252</td>
      <td>SUPRD</td>
      <td>4</td>
      <td>2012-07-09</td>
      <td>2012-08-06</td>
      <td>2012-07-11</td>
      <td>2</td>
      <td>51.300</td>
      <td>Suprêmes délices</td>
      <td>Boulevard Tirou, 255</td>
      <td>Charleroi</td>
      <td>Western Europe</td>
      <td>B-6000</td>
      <td>Belgium</td>
    </tr>
  </tbody>
</table>
</div>



```python
display(order_df.dtypes)
ax = order_df.hist(figsize = (6,9))
```


    Id                  int64
    CustomerId         object
    EmployeeId          int64
    OrderDate          object
    RequiredDate       object
    ShippedDate        object
    ShipVia             int64
    Freight           float64
    ShipName           object
    ShipAddress        object
    ShipCity           object
    ShipRegion         object
    ShipPostalCode     object
    ShipCountry        object
    dtype: object



![png](output_38_1.png)



```python
# Our goal is a marketing report rather than an operations report,  so removing less relevent columns
order_df.drop(columns = ['ShipAddress', 'ShipCity', 'ShipPostalCode', 'Freight' ], inplace=True)
```


```python
order_df['CustomerId'].unique()

```




    array(['VINET', 'TOMSP', 'HANAR', 'VICTE', 'SUPRD', 'CHOPS', 'RICSU',
           'WELLI', 'HILAA', 'ERNSH', 'CENTC', 'OTTIK', 'QUEDE', 'RATTC',
           'FOLKO', 'BLONP', 'WARTH', 'FRANK', 'GROSR', 'WHITC', 'SPLIR',
           'QUICK', 'MAGAA', 'TORTU', 'MORGK', 'BERGS', 'LEHMS', 'ROMEY',
           'LILAS', 'RICAR', 'REGGC', 'BSBEV', 'COMMI', 'TRADH', 'HUNGO',
           'WANDK', 'GODOS', 'OLDWO', 'LONEP', 'ANATR', 'THEBI', 'DUMO',
           'ISLAT', 'PERIC', 'KOENE', 'SAVEA', 'BOLID', 'FURIB', 'BONAP',
           'MEREP', 'PRINI', 'SIMOB', 'FAMIA', 'LAMAI', 'PICCO', 'AROUT',
           'SEVES', 'DRACD', 'EASTC', 'ANTO', 'GALED', 'VAFFE', 'QUEE',
           'WOLZA', 'HUNGC', 'SANTG', 'BOTTM', 'LINOD', 'FOLIG', 'OCEA',
           'FRANS', 'GOURL', 'CONSH', 'RANCH', 'LAZYK', 'LAUGB', 'BLAUS',
           'NORTS', 'CACTU', 'GREAL', 'MAISD', 'TRAIH', 'LETSS', 'WILMK',
           'THECR', 'ALFKI', 'FRANR', 'SPECD', 'LACOR'], dtype=object)



#### 741 Repeat Customers


```python
order_df['CustomerId'].duplicated().value_counts()

```




    True     741
    False     89
    Name: CustomerId, dtype: int64




```python
order_df['ShipName'].duplicated().value_counts()
```




    True     740
    False     90
    Name: ShipName, dtype: int64




```python
order_df['ShipCountry'].duplicated().value_counts()
```




    True     809
    False     21
    Name: ShipCountry, dtype: int64



#### Top Sales People (by # of Orders)

#### Western Europe is the top region for orders (276),  North America (152), and South America (145)

#### United States and Germany are tied for the largest customer base (122)


```python
for col in order_df.columns:
    print(col, "\n", order_df[col].value_counts().head(900), "\n")
```

    Id 
     11077    1
    10520    1
    10530    1
    10529    1
    10528    1
    10527    1
    10526    1
    10525    1
    10524    1
    10523    1
    10522    1
    10521    1
    10519    1
    10532    1
    10518    1
    10517    1
    10516    1
    10515    1
    10514    1
    10513    1
    10512    1
    10511    1
    10510    1
    10509    1
    10531    1
    10533    1
    10507    1
    10546    1
    10556    1
    10555    1
            ..
    10772    1
    10771    1
    10770    1
    10769    1
    10791    1
    10792    1
    10793    1
    10794    1
    10815    1
    10814    1
    10813    1
    10812    1
    10811    1
    10810    1
    10809    1
    10808    1
    10807    1
    10806    1
    10805    1
    10804    1
    10803    1
    10802    1
    10801    1
    10800    1
    10799    1
    10798    1
    10797    1
    10796    1
    10795    1
    10248    1
    Name: Id, Length: 830, dtype: int64 
    
    CustomerId 
     SAVEA    31
    ERNSH    30
    QUICK    28
    FOLKO    19
    HUNGO    19
    BERGS    18
    HILAA    18
    RATTC    18
    BONAP    17
    LEHMS    15
    FRANK    15
    WARTH    15
    WHITC    14
    LAMAI    14
    KOENE    14
    BOTTM    14
    HANAR    14
    LILAS    14
    AROUT    13
    MEREP    13
    QUEE     13
    REGGC    12
    SUPRD    12
    LINOD    12
    BLONP    11
    RICAR    11
    GREAL    11
    VAFFE    11
    WANDK    10
    VICTE    10
             ..
    ALFKI     6
    TRADH     6
    PERIC     6
    CACTU     6
    FOLIG     5
    RANCH     5
    VINET     5
    ROMEY     5
    MORGK     5
    PRINI     5
    COMMI     5
    GALED     5
    OCEA      5
    HUNGC     5
    LETSS     4
    SPECD     4
    THEBI     4
    ANATR     4
    DUMO      4
    LACOR     4
    TRAIH     3
    NORTS     3
    LAUGB     3
    BOLID     3
    FRANR     3
    CONSH     3
    THECR     3
    LAZYK     2
    GROSR     2
    CENTC     1
    Name: CustomerId, Length: 89, dtype: int64 
    
    EmployeeId 
     4    156
    3    127
    1    123
    8    104
    2     96
    7     72
    6     67
    9     43
    5     42
    Name: EmployeeId, dtype: int64 
    
    OrderDate 
     2014-02-26    6
    2014-04-09    4
    2014-04-01    4
    2014-04-17    4
    2014-04-06    4
    2014-03-11    4
    2014-03-03    4
    2014-05-06    4
    2014-04-30    4
    2014-04-22    4
    2014-03-06    4
    2014-04-14    4
    2014-03-27    4
    2014-03-16    4
    2014-03-19    4
    2014-04-27    4
    2014-05-05    4
    2014-03-24    4
    2014-01-29    3
    2013-12-18    3
    2014-02-27    3
    2014-02-04    3
    2014-03-09    3
    2014-01-07    3
    2014-04-29    3
    2014-03-13    3
    2014-03-17    3
    2014-04-15    3
    2014-03-31    3
    2014-02-20    3
                 ..
    2012-09-25    1
    2012-12-19    1
    2012-07-26    1
    2013-01-24    1
    2012-08-02    1
    2012-09-10    1
    2012-07-15    1
    2012-08-12    1
    2013-12-04    1
    2012-10-10    1
    2013-08-26    1
    2012-10-02    1
    2013-06-25    1
    2012-09-13    1
    2012-10-11    1
    2012-11-21    1
    2013-01-31    1
    2012-08-13    1
    2013-08-01    1
    2012-09-16    1
    2013-03-28    1
    2012-09-03    1
    2013-02-26    1
    2012-08-16    1
    2013-10-20    1
    2013-05-15    1
    2012-08-07    1
    2013-05-06    1
    2013-10-15    1
    2013-09-16    1
    Name: OrderDate, Length: 480, dtype: int64 
    
    RequiredDate 
     2014-03-26    7
    2014-04-28    6
    2013-02-13    5
    2014-05-06    5
    2014-05-15    5
    2014-04-08    5
    2014-04-27    5
    2014-03-24    5
    2014-04-17    5
    2012-11-26    4
    2014-02-04    4
    2014-06-03    4
    2014-02-24    4
    2014-05-12    4
    2013-04-08    4
    2014-05-13    4
    2014-05-04    4
    2014-04-24    4
    2014-04-02    4
    2014-03-06    4
    2014-02-06    4
    2014-05-25    4
    2014-04-29    4
    2014-04-16    4
    2014-06-02    4
    2014-04-23    4
    2014-03-20    4
    2014-01-27    4
    2014-03-30    4
    2013-10-20    3
                 ..
    2013-06-30    1
    2013-02-10    1
    2013-10-22    1
    2012-08-28    1
    2012-12-11    1
    2013-01-27    1
    2012-12-13    1
    2013-03-31    1
    2012-10-30    1
    2012-08-08    1
    2013-05-14    1
    2013-01-21    1
    2012-10-15    1
    2013-03-26    1
    2013-02-12    1
    2013-12-08    1
    2013-06-12    1
    2012-11-05    1
    2013-02-05    1
    2012-07-24    1
    2012-11-15    1
    2012-10-09    1
    2012-12-18    1
    2013-03-24    1
    2013-03-20    1
    2012-12-27    1
    2013-04-25    1
    2013-09-26    1
    2013-02-17    1
    2012-09-12    1
    Name: RequiredDate, Length: 454, dtype: int64 
    
    ShippedDate 
     2014-04-10    8
    2014-03-18    7
    2014-05-01    6
    2014-01-23    6
    2014-01-30    6
    2014-04-08    6
    2014-04-24    6
    2014-03-13    5
    2013-11-18    5
    2014-04-02    5
    2014-05-04    5
    2013-11-05    5
    2013-08-01    5
    2013-03-14    5
    2014-01-14    5
    2012-12-02    5
    2014-04-20    5
    2014-03-04    5
    2014-01-05    5
    2014-02-12    5
    2012-12-13    4
    2013-01-27    4
    2014-02-20    4
    2013-07-04    4
    2013-03-03    4
    2014-02-18    4
    2013-07-14    4
    2014-03-27    4
    2014-02-23    4
    2013-12-19    4
                 ..
    2013-06-10    1
    2013-11-11    1
    2012-11-06    1
    2013-06-26    1
    2013-06-16    1
    2013-03-06    1
    2012-09-13    1
    2012-08-14    1
    2013-06-20    1
    2013-02-06    1
    2013-02-18    1
    2013-10-08    1
    2013-02-03    1
    2012-09-25    1
    2013-02-11    1
    2014-02-05    1
    2012-12-16    1
    2012-09-02    1
    2012-11-18    1
    2013-07-11    1
    2013-12-17    1
    2013-06-03    1
    2013-06-25    1
    2013-05-06    1
    2013-11-12    1
    2014-03-10    1
    2013-06-17    1
    2013-05-29    1
    2013-05-26    1
    2013-04-01    1
    Name: ShippedDate, Length: 387, dtype: int64 
    
    ShipVia 
     2    326
    3    255
    1    249
    Name: ShipVia, dtype: int64 
    
    ShipName 
     Save-a-lot Markets                    31
    Ernst Handel                          30
    QUICK-Stop                            28
    Hungry Owl All-Night Grocers          19
    Folk och fä HB                        19
    Rattlesnake Canyon Grocery            18
    HILARION-Abastos                      18
    Berglunds snabbköp                    18
    Bon app'                              17
    Frankenversand                        15
    Lehmanns Marktstand                   15
    Wartian Herkku                        15
    Hanari Carnes                         14
    Bottom-Dollar Markets                 14
    LILA-Supermercado                     14
    White Clover Markets                  14
    Königlich Essen                       14
    La maison d'Asie                      14
    Queen Cozinha                         13
    Around the Horn                       13
    Mère Paillarde                        13
    Reggiani Caseifici                    12
    Suprêmes délices                      12
    LINO-Delicateses                      12
    Ricardo Adocicados                    11
    Vaffeljernet                          11
    Blondel père et fils                  11
    Great Lakes Food Market               11
    Victuailles en stock                  10
    Island Trading                        10
                                          ..
    Pericles Comidas clásicas              6
    Cactus Comidas para llevar             6
    Princesa Isabel Vinhos                 5
    Folies gourmandes                      5
    Morgenstern Gesundkost                 5
    Vins et alcools Chevalier              5
    Galería del gastronómo                 5
    Romero y tomillo                       5
    Rancho grande                          5
    Hungry Coyote Import Store             5
    Comércio Mineiro                       5
    Alfred's Futterkiste                   5
    Océano Atlántico Ltda.                 5
    The Big Cheese                         4
    Du monde entier                        4
    Let's Stop N Shop                      4
    Spécialités du monde                   4
    Ana Trujillo Emparedados y helados     4
    La corne d'abondance                   4
    North/South                            3
    Trail's Head Gourmet Provisioners      3
    Bólido Comidas preparadas              3
    France restauration                    3
    The Cracker Box                        3
    Consolidated Holdings                  3
    Laughing Bacchus Wine Cellars          3
    GROSELLA-Restaurante                   2
    Lazy K Kountry Store                   2
    Alfreds Futterkiste                    1
    Centro comercial Moctezuma             1
    Name: ShipName, Length: 90, dtype: int64 
    
    ShipRegion 
     Western Europe     276
    North America      152
    South America      145
    British Isles       75
    Southern Europe     64
    Northern Europe     55
    Scandinavia         28
    Central America     28
    Eastern Europe       7
    Name: ShipRegion, dtype: int64 
    
    ShipCountry 
     USA            122
    Germany        122
    Brazil          83
    France          77
    UK              56
    Venezuela       46
    Austria         40
    Sweden          37
    Canada          30
    Italy           28
    Mexico          28
    Spain           23
    Finland         22
    Belgium         19
    Ireland         19
    Denmark         18
    Switzerland     18
    Argentina       16
    Portugal        13
    Poland           7
    Norway           6
    Name: ShipCountry, dtype: int64 
    
    

# Customer Demographic tables do not exist


```python
select_data = 'SELECT * FROM CustomerDemographic;'
customer_demographic_df = load_data(select_data)
```

    Table not loaded
    


```python
select_data = 'SELECT * FROM CustomerCustomerDemo LIMIT 5;'
ccustomer_demo_df = load_data(select_data)


```

    Table not loaded
    

# Defined Functions


```python
# Define generic functions to do simple statistics evaluations on 2 sample comparisons between features and metrics
# The purpose is to see if specific feature has a significant effect on a metric (example feature : discount)
# An example of the metric is "Order Quantity" or "Total Product Spend ($)"
# This is an example of univariate analysis where we check for the significance of a single feature while all other
# effects are not taken into account 

def two_sample_tests(df, feature, metric, alpha = 0.05): 
    
    ''' Run standard tests for two samples to check feature impact on outcomes / metrics
        Uses get_group to create populations
         
        Includes: 
          - Test for normality of the samples
          - Equivalence of populations 
                  - MannWhitneyU for non-normal populations
                  - 2-sided T-Tests for normal populations
          - Equivalence of the population in terms of mean and variance using Levene test
    
    Args:
        feature: these are 
        metric: these are scalar quantities that are desired results, such as order quantity, product revenue, 
            customer spend, etc.
        alpha is compared to the p-value analysed by scipy or statsmodel, and defines the percentage that the result is due to chance. 
            alpha typically = 0.05,  or 5%
              
        
    Returns:
        group1 and group2 samples for plotting
        
    Ex:
    >> no_discount, with_discount = two_sample_tests(orderdetail_df, 'Discounts', 'Quantity')
    
    '''
    group1 = df.groupby(feature).get_group(0)[metric]
    group2 = df.groupby(feature).get_group(1)[metric]
    
    print(f'Number of datapoints in group 1, no {feature} = {len(group1)}')
    print(f'Number of datapoints in group 2, with {feature} = {len(group2)}')
       
    if (len(group1) or len(group2)) > 20:
        print("Number of datapoints in the samples > 30, possible to assume normality through the Central Limit Theorem", '\n')
        print("Check the normality anyway!")    
        
# Run the test for a normal distribution        
    normtest_p1, normtest_p2 = normdist_test(group1, group2, alpha)

# Check for the equivalence of the populations in terms of mean and variance, using Levene test, 
# Then check the equivalence of the populations using either MannWhitneyU or the 2-tailed T-test

    levene_test(group1, group2, alpha)
    
    if (normtest_p1 or normtest_p2) < alpha:
        print("Need to use the non-parametric test (Mann-WhitneyU) to determine the equivalence of the populations", '\n')
        mannwhitney_test(group1, group2, alpha)
        
    else:
        print("Since the distributions are normal, proceed with the 2 sided t-test")
        t_test(group1, group2, alpha)
        
    return group1, group2


def normdist_test(group1, group2, alpha): 
    
    sns.distplot(group1)
    plt.show()
    stat_test1, normtest_p1 = stat.normaltest(group1)
    
    if normtest_p1 < alpha:
        print(f"Sample 1 P-value {normtest_p1} is less than alpha {alpha}")
        print(f"Sample 1 Normality: We can reject the null hypothesis that the 1st sample comes from a normal distribution", '\n')
    else:
        print("Sample 1 Normality: We can accept the null hypothesis that the 1st sample comes from a normal distribution", '\n')
    
    sns.distplot(group2)
    plt.show()
    
    stat_test2, normtest_p2 = stat.normaltest(group2)
    
    if normtest_p2 < alpha:
        print(f"Sample 2 P-value {normtest_p1} is less than alpha {alpha}")
        print(f"Sample 2 Normality: We can reject the null hypothesis that the 2nd sample comes from a normal distribution", '\n')
    else:
        print("Sample 2 Normality: We can accept the null hypothesis that the 2nd samples comes from a normal distribution", '\n')
        
    return normtest_p1, normtest_p2
        
def mannwhitney_test(group1, group2, alpha):
    mw_stat, mw_p = stat.mannwhitneyu(group1, group2)
    print(f"MWU: The U statistic is {mw_stat}, and the p-value is {mw_p}")
    if mw_p < alpha:
        print(f"MWU: The MannWhitneyU p-value {mw_p} is less than alpha {alpha}")
        print("MannWhitneyU test: We can reject the null hypothesis that the two populations are equal",'\n')
    else:
        print("MannWhitneyU test: We can accept the null hypothesis that the two populations are equal",'\n')
        
    return 
          
# This is a Two-tailed T-Test,  whether two groups are equivalent to one another
# https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/hypothesis-testing/one-tailed-test-or-two/

def t_test(group1, group2, alpha):  
    # Usage: t_test
    ttest_stat, ttest_p = stat.ttest_ind(group1, group2)
    if ttest_p < alpha:
        print(f"The 2-tailed T-test p value {ttest_p}, is less than alpha {alpha}")
        print("The 2-tailed T-test: We reject the null hypothesis that the populations have identical variance",'\n')
    else:
        print("The 2-tailed T-test: We accept the null hypthesis that the populations have identical variance",'\n')

        
    return
          
          
def levene_test(group1, group2, alpha):
    levene_stat, levene_p = stat.levene(group1, group2)
    if levene_p < alpha:
        print(f"The Levene p-value, {levene_p} is less than alpha {alpha}")
        print("Levene test: We reject the null hypothesis that the variances are equal",'\n')
    else:
        print("Levene test: We accept the null hypothesis that the variances of the populations equal", '\n')
        
    return

#Plot the data, with standard error of measurement (sem)

def plot_two_sample_tests(group1, group2, feature, metric):
    '''Plot group1 and group2 next to each other along with the sem (standard error measurement)'''
    pd.options.display.float_format = '{:,.2f}'.format
    print(f'The average {metric} sold, without {feature} is {group1.mean()}')
    print(f'The average {metric} sold, with {feature} is {group2.mean()}')
    
    plt.bar(x=f'Without {feature}', height = group1.mean(), yerr = stat.sem(group1))
    plt.bar(x=f'With {feature}', height = group2.mean(), yerr = stat.sem(group2))
    
    plt.title(f'Effect of {feature} on {metric}')
    plt.ylabel(f'Average {metric}');
    
    return         

```

# Hypothesis 1: Does discount have a significant impact on the quantity ordered?

## The null hypothesis H0 is that the discount has no significant impact on quantity ordered.
## The alternative hypthesis H1 is that discount does have a significant impact on quantity ordered.


```python
# Use the functions built to perform the analysis automatically

no_discount, with_discount = two_sample_tests(orderdetail_df, 'Discounts', 'Quantity')
plot_two_sample_tests(no_discount, with_discount, "Discounts", "Order Quantity")

```

    Number of datapoints in group 1, no Discounts = 1317
    Number of datapoints in group 2, with Discounts = 838
    Number of datapoints in the samples > 30, possible to assume normality through the Central Limit Theorem 
    
    Check the normality anyway!
    


![png](output_53_1.png)


    Sample 1 P-value 5.579637380545965e-119 is less than alpha 0.05
    Sample 1 Normality: We can reject the null hypothesis that the 1st sample comes from a normal distribution 
    
    


![png](output_53_3.png)


    Sample 2 P-value 5.579637380545965e-119 is less than alpha 0.05
    Sample 2 Normality: We can reject the null hypothesis that the 2nd sample comes from a normal distribution 
    
    The Levene p-value, 0.00012091378376079568 is less than alpha 0.05
    Levene test: We reject the null hypothesis that the variances are equal 
    
    Need to use the non-parametric test (Mann-WhitneyU) to determine the equivalence of the populations 
    
    MWU: The U statistic is 461541.0, and the p-value is 6.629381826999866e-11
    MWU: The MannWhitneyU p-value 6.629381826999866e-11 is less than alpha 0.05
    MannWhitneyU test: We can reject the null hypothesis that the two populations are equal 
    
    The average Order Quantity sold, without Discounts is 21.715261958997722
    The average Order Quantity sold, with Discounts is 27.10978520286396
    


![png](output_53_5.png)


## Calculate which Levels of discount are significant 


```python
# The lack of overlap of confidence intervals agrees with the above test
# The difference in average quantity not due to random chance of 5% or more

# Now calculate which levels of discount are significant towards increasing the quantity
# Use of Tukey's test

data = orderdetail_df['Quantity'].values
labels = orderdetail_df['Discount'].values

print(data), print(labels)

import statsmodels.api as sms

model = sms.stats.multicomp.pairwise_tukeyhsd(data,labels)
model.summary()

```

    [12 10  5 ...  2  4  2]
    [0.   0.   0.   ... 0.01 0.   0.  ]
    




<table class="simpletable">
<caption>Multiple Comparison of Means - Tukey HSD, FWER=0.05</caption>
<tr>
  <th>group1</th> <th>group2</th> <th>meandiff</th>  <th>p-adj</th>   <th>lower</th>   <th>upper</th>  <th>reject</th>
</tr>
<tr>
    <td>0.0</td>   <td>0.01</td>  <td>-19.7153</td>   <td>0.9</td>  <td>-80.3306</td> <td>40.9001</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.02</td>  <td>-19.7153</td>   <td>0.9</td>   <td>-62.593</td> <td>23.1625</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.03</td>  <td>-20.0486</td>  <td>0.725</td> <td>-55.0714</td> <td>14.9742</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.04</td>  <td>-20.7153</td>   <td>0.9</td>  <td>-81.3306</td> <td>39.9001</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.05</td>   <td>6.2955</td>  <td>0.0011</td>  <td>1.5381</td>  <td>11.053</td>   <td>True</td> 
</tr>
<tr>
    <td>0.0</td>   <td>0.06</td>  <td>-19.7153</td>   <td>0.9</td>  <td>-80.3306</td> <td>40.9001</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>    <td>0.1</td>   <td>3.5217</td>  <td>0.4269</td>  <td>-1.3783</td> <td>8.4217</td>   <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.15</td>   <td>6.6669</td>  <td>0.0014</td>   <td>1.551</td>  <td>11.7828</td>  <td>True</td> 
</tr>
<tr>
    <td>0.0</td>    <td>0.2</td>   <td>5.3096</td>  <td>0.0303</td>  <td>0.2508</td>  <td>10.3684</td>  <td>True</td> 
</tr>
<tr>
    <td>0.0</td>   <td>0.25</td>    <td>6.525</td>  <td>0.0023</td>  <td>1.3647</td>  <td>11.6852</td>  <td>True</td> 
</tr>
<tr>
   <td>0.01</td>   <td>0.02</td>     <td>0.0</td>     <td>0.9</td>  <td>-74.2101</td> <td>74.2101</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.03</td>   <td>-0.3333</td>   <td>0.9</td>  <td>-70.2993</td> <td>69.6326</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.04</td>    <td>-1.0</td>     <td>0.9</td>  <td>-86.6905</td> <td>84.6905</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.05</td>   <td>26.0108</td>   <td>0.9</td>   <td>-34.745</td> <td>86.7667</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.06</td>     <td>0.0</td>     <td>0.9</td>  <td>-85.6905</td> <td>85.6905</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>    <td>0.1</td>   <td>23.237</td>    <td>0.9</td>  <td>-37.5302</td> <td>84.0042</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.15</td>   <td>26.3822</td>   <td>0.9</td>  <td>-34.4028</td> <td>87.1671</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>    <td>0.2</td>   <td>25.0248</td>   <td>0.9</td>  <td>-35.7554</td> <td>85.805</td>   <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.25</td>   <td>26.2403</td>   <td>0.9</td>  <td>-34.5485</td> <td>87.029</td>   <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.03</td>   <td>-0.3333</td>   <td>0.9</td>  <td>-55.6463</td> <td>54.9796</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.04</td>    <td>-1.0</td>     <td>0.9</td>  <td>-75.2101</td> <td>73.2101</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.05</td>   <td>26.0108</td> <td>0.6622</td> <td>-17.0654</td> <td>69.087</td>   <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.06</td>     <td>0.0</td>     <td>0.9</td>  <td>-74.2101</td> <td>74.2101</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>    <td>0.1</td>   <td>23.237</td>  <td>0.7914</td> <td>-19.8552</td> <td>66.3292</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.15</td>   <td>26.3822</td> <td>0.6461</td> <td>-16.7351</td> <td>69.4994</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>    <td>0.2</td>   <td>25.0248</td> <td>0.7089</td> <td>-18.0857</td> <td>68.1354</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.25</td>   <td>26.2403</td> <td>0.6528</td> <td>-16.8823</td> <td>69.3628</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.04</td>   <td>-0.6667</td>   <td>0.9</td>  <td>-70.6326</td> <td>69.2993</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.05</td>   <td>26.3441</td> <td>0.3639</td>  <td>-8.9214</td> <td>61.6096</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.06</td>   <td>0.3333</td>    <td>0.9</td>  <td>-69.6326</td> <td>70.2993</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>    <td>0.1</td>   <td>23.5703</td> <td>0.5338</td> <td>-11.7147</td> <td>58.8553</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.15</td>   <td>26.7155</td> <td>0.3436</td>  <td>-8.6001</td> <td>62.0311</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>    <td>0.2</td>   <td>25.3582</td>  <td>0.428</td>  <td>-9.9492</td> <td>60.6656</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.25</td>   <td>26.5736</td> <td>0.3525</td>  <td>-8.7485</td> <td>61.8957</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.05</td>   <td>27.0108</td>   <td>0.9</td>   <td>-33.745</td> <td>87.7667</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.06</td>     <td>1.0</td>     <td>0.9</td>  <td>-84.6905</td> <td>86.6905</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>    <td>0.1</td>   <td>24.237</td>    <td>0.9</td>  <td>-36.5302</td> <td>85.0042</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.15</td>   <td>27.3822</td>   <td>0.9</td>  <td>-33.4028</td> <td>88.1671</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>    <td>0.2</td>   <td>26.0248</td>   <td>0.9</td>  <td>-34.7554</td> <td>86.805</td>   <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.25</td>   <td>27.2403</td>   <td>0.9</td>  <td>-33.5485</td> <td>88.029</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>   <td>0.06</td>  <td>-26.0108</td>   <td>0.9</td>  <td>-86.7667</td> <td>34.745</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>    <td>0.1</td>   <td>-2.7738</td>   <td>0.9</td>   <td>-9.1822</td> <td>3.6346</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>   <td>0.15</td>   <td>0.3714</td>    <td>0.9</td>   <td>-6.2036</td> <td>6.9463</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>    <td>0.2</td>   <td>-0.986</td>    <td>0.9</td>   <td>-7.5166</td> <td>5.5447</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>   <td>0.25</td>   <td>0.2294</td>    <td>0.9</td>   <td>-6.3801</td>  <td>6.839</td>   <td>False</td>
</tr>
<tr>
   <td>0.06</td>    <td>0.1</td>   <td>23.237</td>    <td>0.9</td>  <td>-37.5302</td> <td>84.0042</td>  <td>False</td>
</tr>
<tr>
   <td>0.06</td>   <td>0.15</td>   <td>26.3822</td>   <td>0.9</td>  <td>-34.4028</td> <td>87.1671</td>  <td>False</td>
</tr>
<tr>
   <td>0.06</td>    <td>0.2</td>   <td>25.0248</td>   <td>0.9</td>  <td>-35.7554</td> <td>85.805</td>   <td>False</td>
</tr>
<tr>
   <td>0.06</td>   <td>0.25</td>   <td>26.2403</td>   <td>0.9</td>  <td>-34.5485</td> <td>87.029</td>   <td>False</td>
</tr>
<tr>
    <td>0.1</td>   <td>0.15</td>   <td>3.1452</td>    <td>0.9</td>   <td>-3.5337</td>  <td>9.824</td>   <td>False</td>
</tr>
<tr>
    <td>0.1</td>    <td>0.2</td>   <td>1.7879</td>    <td>0.9</td>   <td>-4.8474</td> <td>8.4231</td>   <td>False</td>
</tr>
<tr>
    <td>0.1</td>   <td>0.25</td>   <td>3.0033</td>    <td>0.9</td>   <td>-3.7096</td> <td>9.7161</td>   <td>False</td>
</tr>
<tr>
   <td>0.15</td>    <td>0.2</td>   <td>-1.3573</td>   <td>0.9</td>   <td>-8.1536</td> <td>5.4389</td>   <td>False</td>
</tr>
<tr>
   <td>0.15</td>   <td>0.25</td>   <td>-0.1419</td>   <td>0.9</td>   <td>-7.014</td>  <td>6.7302</td>   <td>False</td>
</tr>
<tr>
    <td>0.2</td>   <td>0.25</td>   <td>1.2154</td>    <td>0.9</td>   <td>-5.6143</td> <td>8.0451</td>   <td>False</td>
</tr>
</table>




```python
model.plot_simultaneous()
plt.ylabel('Discount')
plt.xlabel('Average Order Quantity');
```


![png](output_56_0.png)


## Hypothesis 1 Results

### Reject the null hypothesis that discounted and undiscounted products have similar quantities

### Only 5%, 15%, 20%, and 25% discounts have significant impact on the average order quantity

### Interestingly a 10% discount does not appear to have a significant effect relative to no discount


# Hypothesis 2:  Does discount impact the total amount a customer will spend (Total Product Spend) ?


```python
# Use the functions built to perform the analysis automatically

no_disc_rev_impact, with_disc_rev_impact = two_sample_tests(orderdetail_df, 'Discounts', 'Total Product Spend ($)')

plot_two_sample_tests(no_disc_rev_impact, with_disc_rev_impact, 'Discounts', 'Total Product Spend ($) ')
```

    Number of datapoints in group 1, no Discounts = 1317
    Number of datapoints in group 2, with Discounts = 838
    Number of datapoints in the samples > 30, possible to assume normality through the Central Limit Theorem 
    
    Check the normality anyway!
    


![png](output_59_1.png)


    Sample 1 P-value 0.0 is less than alpha 0.05
    Sample 1 Normality: We can reject the null hypothesis that the 1st sample comes from a normal distribution 
    
    


![png](output_59_3.png)


    Sample 2 P-value 0.0 is less than alpha 0.05
    Sample 2 Normality: We can reject the null hypothesis that the 2nd sample comes from a normal distribution 
    
    Levene test: We accept the null hypothesis that the variances of the populations equal 
    
    Need to use the non-parametric test (Mann-WhitneyU) to determine the equivalence of the populations 
    
    MWU: The U statistic is 525843.5, and the p-value is 0.03252534474485538
    MWU: The MannWhitneyU p-value 0.03252534474485538 is less than alpha 0.05
    MannWhitneyU test: We can reject the null hypothesis that the two populations are equal 
    
    The average Total Product Spend ($)  sold, without Discounts is 570.0065375854215
    The average Total Product Spend ($)  sold, with Discounts is 614.671156921241
    


![png](output_59_5.png)


## The overlap of confidence intervals agrees with the above test, that the impact of discount on product revenue is not significant

## We aren't able to reject the null hypothesis.  There effect of discounts on customer spending appears to be insignificant.

## No need to test whether specific discounts might have impact on the product revenue



```python
# Tukey tests show that none of the different discounts has a meaningful impact on the Total Product Spend
data = orderdetail_df['Total Product Spend ($)'].values 
labels = orderdetail_df['Discount'].values

print(data), print(labels)

import statsmodels.api as sms

model = sms.stats.multicomp.pairwise_tukeyhsd(data,labels) 
model.summary()
```

    [168.   98.  174.  ...  29.7  31.   26. ]
    [0.   0.   0.   ... 0.01 0.   0.  ]
    




<table class="simpletable">
<caption>Multiple Comparison of Means - Tukey HSD, FWER=0.05</caption>
<tr>
  <th>group1</th> <th>group2</th> <th>meandiff</th>   <th>p-adj</th>    <th>lower</th>     <th>upper</th>   <th>reject</th>
</tr>
<tr>
    <td>0.0</td>   <td>0.01</td>  <td>-540.3065</td>   <td>0.9</td>  <td>-3662.2549</td> <td>2581.6418</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.02</td>  <td>-540.1165</td>   <td>0.9</td>  <td>-2748.5047</td> <td>1668.2716</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.03</td>  <td>-529.703</td>    <td>0.9</td>  <td>-2333.5278</td> <td>1274.1217</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.04</td>  <td>-492.2465</td>   <td>0.9</td>  <td>-3614.1949</td> <td>2629.7018</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.05</td>  <td>227.9252</td>  <td>0.0954</td>  <td>-17.1036</td>   <td>472.954</td>   <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.06</td>  <td>-506.0865</td>   <td>0.9</td>  <td>-3628.0349</td> <td>2615.8618</td>  <td>False</td>
</tr>
<tr>
    <td>0.0</td>    <td>0.1</td>  <td>-41.1098</td>    <td>0.9</td>    <td>-293.48</td>  <td>211.2604</td>   <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.15</td>  <td>-12.6424</td>    <td>0.9</td>   <td>-276.1341</td> <td>250.8493</td>   <td>False</td>
</tr>
<tr>
    <td>0.0</td>    <td>0.2</td>  <td>-16.0866</td>    <td>0.9</td>   <td>-276.6374</td> <td>244.4641</td>   <td>False</td>
</tr>
<tr>
    <td>0.0</td>   <td>0.25</td>   <td>72.4517</td>    <td>0.9</td>   <td>-193.3232</td> <td>338.2266</td>   <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.02</td>    <td>0.19</td>      <td>0.9</td>  <td>-3821.9494</td> <td>3822.3294</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.03</td>   <td>10.6035</td>    <td>0.9</td>  <td>-3592.9441</td> <td>3614.1511</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.04</td>    <td>48.06</td>     <td>0.9</td>  <td>-4365.3664</td> <td>4461.4864</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.05</td>  <td>768.2318</td>    <td>0.9</td>  <td>-2360.9551</td> <td>3897.4187</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.06</td>    <td>34.22</td>     <td>0.9</td>  <td>-4379.2064</td> <td>4447.6464</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>    <td>0.1</td>  <td>499.1968</td>    <td>0.9</td>  <td>-2630.5736</td> <td>3628.9671</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.15</td>  <td>527.6642</td>    <td>0.9</td>  <td>-2603.0226</td> <td>3658.3509</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>    <td>0.2</td>  <td>524.2199</td>    <td>0.9</td>  <td>-2606.2207</td> <td>3654.6605</td>  <td>False</td>
</tr>
<tr>
   <td>0.01</td>   <td>0.25</td>  <td>612.7582</td>    <td>0.9</td>  <td>-2518.1215</td> <td>3743.638</td>   <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.03</td>   <td>10.4135</td>    <td>0.9</td>   <td>-2838.441</td> <td>2859.268</td>   <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.04</td>    <td>47.87</td>     <td>0.9</td>  <td>-3774.2694</td> <td>3870.0094</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.05</td>  <td>768.0418</td>    <td>0.9</td>  <td>-1450.5676</td> <td>2986.6511</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.06</td>    <td>34.03</td>     <td>0.9</td>  <td>-3788.1094</td> <td>3856.1694</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>    <td>0.1</td>  <td>499.0068</td>    <td>0.9</td>  <td>-1720.4254</td> <td>2718.4389</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.15</td>  <td>527.4742</td>    <td>0.9</td>  <td>-1693.2501</td> <td>2748.1984</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>    <td>0.2</td>  <td>524.0299</td>    <td>0.9</td>  <td>-1696.3473</td> <td>2744.4071</td>  <td>False</td>
</tr>
<tr>
   <td>0.02</td>   <td>0.25</td>  <td>612.5682</td>    <td>0.9</td>  <td>-1608.4281</td> <td>2833.5645</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.04</td>   <td>37.4565</td>    <td>0.9</td>  <td>-3566.0911</td> <td>3641.0041</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.05</td>  <td>757.6283</td>    <td>0.9</td>  <td>-1058.6958</td> <td>2573.9523</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.06</td>   <td>23.6165</td>    <td>0.9</td>  <td>-3579.9311</td> <td>3627.1641</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>    <td>0.1</td>  <td>488.5933</td>    <td>0.9</td>  <td>-1328.7357</td> <td>2305.9222</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.15</td>  <td>517.0607</td>    <td>0.9</td>  <td>-1301.8461</td> <td>2335.9674</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>    <td>0.2</td>  <td>513.6164</td>    <td>0.9</td>  <td>-1304.8666</td> <td>2332.0994</td>  <td>False</td>
</tr>
<tr>
   <td>0.03</td>   <td>0.25</td>  <td>602.1547</td>    <td>0.9</td>  <td>-1217.0842</td> <td>2421.3936</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.05</td>  <td>720.1718</td>    <td>0.9</td>  <td>-2409.0151</td> <td>3849.3587</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.06</td>   <td>-13.84</td>     <td>0.9</td>  <td>-4427.2664</td> <td>4399.5864</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>    <td>0.1</td>  <td>451.1368</td>    <td>0.9</td>  <td>-2678.6336</td> <td>3580.9071</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.15</td>  <td>479.6042</td>    <td>0.9</td>  <td>-2651.0826</td> <td>3610.2909</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>    <td>0.2</td>  <td>476.1599</td>    <td>0.9</td>  <td>-2654.2807</td> <td>3606.6005</td>  <td>False</td>
</tr>
<tr>
   <td>0.04</td>   <td>0.25</td>  <td>564.6982</td>    <td>0.9</td>  <td>-2566.1815</td> <td>3695.578</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>   <td>0.06</td>  <td>-734.0118</td>   <td>0.9</td>  <td>-3863.1987</td> <td>2395.1751</td>  <td>False</td>
</tr>
<tr>
   <td>0.05</td>    <td>0.1</td>  <td>-269.035</td>  <td>0.2357</td>  <td>-599.0955</td>  <td>61.0255</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>   <td>0.15</td>  <td>-240.5676</td> <td>0.4454</td>  <td>-579.2076</td>  <td>98.0724</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>    <td>0.2</td>  <td>-244.0119</td> <td>0.4118</td>  <td>-580.3686</td>  <td>92.3449</td>   <td>False</td>
</tr>
<tr>
   <td>0.05</td>   <td>0.25</td>  <td>-155.4735</td>   <td>0.9</td>   <td>-495.8931</td>  <td>184.946</td>   <td>False</td>
</tr>
<tr>
   <td>0.06</td>    <td>0.1</td>  <td>464.9768</td>    <td>0.9</td>  <td>-2664.7936</td> <td>3594.7471</td>  <td>False</td>
</tr>
<tr>
   <td>0.06</td>   <td>0.15</td>  <td>493.4442</td>    <td>0.9</td>  <td>-2637.2426</td> <td>3624.1309</td>  <td>False</td>
</tr>
<tr>
   <td>0.06</td>    <td>0.2</td>  <td>489.9999</td>    <td>0.9</td>  <td>-2640.4407</td> <td>3620.4405</td>  <td>False</td>
</tr>
<tr>
   <td>0.06</td>   <td>0.25</td>  <td>578.5382</td>    <td>0.9</td>  <td>-2552.3415</td> <td>3709.418</td>   <td>False</td>
</tr>
<tr>
    <td>0.1</td>   <td>0.15</td>   <td>28.4674</td>    <td>0.9</td>   <td>-315.5219</td> <td>372.4568</td>   <td>False</td>
</tr>
<tr>
    <td>0.1</td>    <td>0.2</td>   <td>25.0231</td>    <td>0.9</td>   <td>-316.7187</td>  <td>366.765</td>   <td>False</td>
</tr>
<tr>
    <td>0.1</td>   <td>0.25</td>  <td>113.5615</td>    <td>0.9</td>   <td>-232.1799</td> <td>459.3029</td>   <td>False</td>
</tr>
<tr>
   <td>0.15</td>    <td>0.2</td>   <td>-3.4443</td>    <td>0.9</td>   <td>-353.4794</td> <td>346.5909</td>   <td>False</td>
</tr>
<tr>
   <td>0.15</td>   <td>0.25</td>   <td>85.0941</td>    <td>0.9</td>   <td>-268.847</td>  <td>439.0351</td>   <td>False</td>
</tr>
<tr>
    <td>0.2</td>   <td>0.25</td>   <td>88.5383</td>    <td>0.9</td>   <td>-263.2188</td> <td>440.2954</td>   <td>False</td>
</tr>
</table>



## Summary:  Discounts have no significant effect on customers' product spend



```python
model.plot_simultaneous()
plt.ylabel('Discount')
plt.xlabel('Average Product Spend ($)');
```


![png](output_63_0.png)




# Hypothesis 3 : What is the impact of "Net Discount" on the Quantity?

## Joining the orderdetail and product tables shows that some products were sold with a price lower than the MSRP (Manufacturers Suggested Retail Price),  and some had additional discounts

## When we look at the joined tables below, we see there are actually 4 different pricing strategies:  MSRP No Discounts, MSRP with Discounts, MSRP Reduction,  MSRP Reduction with Discounts


## Create new categorical column "MSRP"  (1 = MSRP price, 0 = MSRP Reduction)
## Create new categorical column "net_discount" (1 = has some kind of discount, 0 = no discounts)

display(orderdetail_df.head())
display(product_df.head(11))


```python
#select_data = 'SELECT * FROM OrderDetail od JOIN "Order" ON od.OrderId = "Order".Id JOIN Product p ON od.ProductId = p.Id;'
#ordetail_ord_prod_df = load_data(select_data)
#ordetail_ord_prod_df.head()

# Use pandas to join orderdetail_df to product_df to take advantage of the added columns in orderdetails_df and because Pandas 
# creates unique column names
order_product_df = orderdetail_df.merge(product_df, left_on='ProductId', right_on='Id')

# Create a new column called "Net Discounts" and a categorical net_discount
order_product_df['Net Discounts'] = 1 - (order_product_df['Final Price'] / order_product_df['UnitPrice_y'])
order_product_df['net_discount'] = np.where(order_product_df['Net Discounts'] == 0,0,1)

# Create a new column MSRP Discount and cateogorical column that calculates the discount based on price only
# The categorical checks whether the order is using the MSRP or not
order_product_df['MSRP Discount'] = 1 - (order_product_df['UnitPrice_x'] / order_product_df['UnitPrice_y'])
order_product_df['MSRP'] = np.where(order_product_df["UnitPrice_x"] == order_product_df["UnitPrice_y"], 1, 0)

order_product_df.head(50)


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id_x</th>
      <th>OrderId</th>
      <th>ProductId</th>
      <th>UnitPrice_x</th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>Discounts</th>
      <th>Final Price</th>
      <th>Total Product Spend ($)</th>
      <th>Id_y</th>
      <th>ProductName</th>
      <th>SupplierId</th>
      <th>CategoryId</th>
      <th>QuantityPerUnit</th>
      <th>UnitPrice_y</th>
      <th>Net Discounts</th>
      <th>net_discount</th>
      <th>MSRP Discount</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10248/11</td>
      <td>10248</td>
      <td>11</td>
      <td>14.00</td>
      <td>12</td>
      <td>0.00</td>
      <td>0</td>
      <td>14.00</td>
      <td>168.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.33</td>
      <td>1</td>
      <td>0.33</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10296/11</td>
      <td>10296</td>
      <td>11</td>
      <td>16.80</td>
      <td>12</td>
      <td>0.00</td>
      <td>0</td>
      <td>16.80</td>
      <td>201.60</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.20</td>
      <td>1</td>
      <td>0.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10327/11</td>
      <td>10327</td>
      <td>11</td>
      <td>16.80</td>
      <td>50</td>
      <td>0.20</td>
      <td>1</td>
      <td>13.44</td>
      <td>672.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.36</td>
      <td>1</td>
      <td>0.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10353/11</td>
      <td>10353</td>
      <td>11</td>
      <td>16.80</td>
      <td>12</td>
      <td>0.20</td>
      <td>1</td>
      <td>13.44</td>
      <td>161.28</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.36</td>
      <td>1</td>
      <td>0.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10365/11</td>
      <td>10365</td>
      <td>11</td>
      <td>16.80</td>
      <td>24</td>
      <td>0.00</td>
      <td>0</td>
      <td>16.80</td>
      <td>403.20</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.20</td>
      <td>1</td>
      <td>0.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10407/11</td>
      <td>10407</td>
      <td>11</td>
      <td>16.80</td>
      <td>30</td>
      <td>0.00</td>
      <td>0</td>
      <td>16.80</td>
      <td>504.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.20</td>
      <td>1</td>
      <td>0.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10434/11</td>
      <td>10434</td>
      <td>11</td>
      <td>16.80</td>
      <td>6</td>
      <td>0.00</td>
      <td>0</td>
      <td>16.80</td>
      <td>100.80</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.20</td>
      <td>1</td>
      <td>0.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10442/11</td>
      <td>10442</td>
      <td>11</td>
      <td>16.80</td>
      <td>30</td>
      <td>0.00</td>
      <td>0</td>
      <td>16.80</td>
      <td>504.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.20</td>
      <td>1</td>
      <td>0.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10443/11</td>
      <td>10443</td>
      <td>11</td>
      <td>16.80</td>
      <td>6</td>
      <td>0.20</td>
      <td>1</td>
      <td>13.44</td>
      <td>80.64</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.36</td>
      <td>1</td>
      <td>0.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10466/11</td>
      <td>10466</td>
      <td>11</td>
      <td>16.80</td>
      <td>10</td>
      <td>0.00</td>
      <td>0</td>
      <td>16.80</td>
      <td>168.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.20</td>
      <td>1</td>
      <td>0.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10486/11</td>
      <td>10486</td>
      <td>11</td>
      <td>16.80</td>
      <td>5</td>
      <td>0.00</td>
      <td>0</td>
      <td>16.80</td>
      <td>84.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.20</td>
      <td>1</td>
      <td>0.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>10489/11</td>
      <td>10489</td>
      <td>11</td>
      <td>16.80</td>
      <td>15</td>
      <td>0.25</td>
      <td>1</td>
      <td>12.60</td>
      <td>189.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.40</td>
      <td>1</td>
      <td>0.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>10528/11</td>
      <td>10528</td>
      <td>11</td>
      <td>21.00</td>
      <td>3</td>
      <td>0.00</td>
      <td>0</td>
      <td>21.00</td>
      <td>63.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>10535/11</td>
      <td>10535</td>
      <td>11</td>
      <td>21.00</td>
      <td>50</td>
      <td>0.10</td>
      <td>1</td>
      <td>18.90</td>
      <td>945.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.10</td>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>10542/11</td>
      <td>10542</td>
      <td>11</td>
      <td>21.00</td>
      <td>15</td>
      <td>0.05</td>
      <td>1</td>
      <td>19.95</td>
      <td>299.25</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.05</td>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>10545/11</td>
      <td>10545</td>
      <td>11</td>
      <td>21.00</td>
      <td>10</td>
      <td>0.00</td>
      <td>0</td>
      <td>21.00</td>
      <td>210.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>10553/11</td>
      <td>10553</td>
      <td>11</td>
      <td>21.00</td>
      <td>15</td>
      <td>0.00</td>
      <td>0</td>
      <td>21.00</td>
      <td>315.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>10566/11</td>
      <td>10566</td>
      <td>11</td>
      <td>21.00</td>
      <td>35</td>
      <td>0.15</td>
      <td>1</td>
      <td>17.85</td>
      <td>624.75</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.15</td>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>10570/11</td>
      <td>10570</td>
      <td>11</td>
      <td>21.00</td>
      <td>15</td>
      <td>0.05</td>
      <td>1</td>
      <td>19.95</td>
      <td>299.25</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.05</td>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>10614/11</td>
      <td>10614</td>
      <td>11</td>
      <td>21.00</td>
      <td>14</td>
      <td>0.00</td>
      <td>0</td>
      <td>21.00</td>
      <td>294.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10637/11</td>
      <td>10637</td>
      <td>11</td>
      <td>21.00</td>
      <td>10</td>
      <td>0.00</td>
      <td>0</td>
      <td>21.00</td>
      <td>210.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>10698/11</td>
      <td>10698</td>
      <td>11</td>
      <td>21.00</td>
      <td>15</td>
      <td>0.00</td>
      <td>0</td>
      <td>21.00</td>
      <td>315.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>10726/11</td>
      <td>10726</td>
      <td>11</td>
      <td>21.00</td>
      <td>5</td>
      <td>0.00</td>
      <td>0</td>
      <td>21.00</td>
      <td>105.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>10770/11</td>
      <td>10770</td>
      <td>11</td>
      <td>21.00</td>
      <td>15</td>
      <td>0.25</td>
      <td>1</td>
      <td>15.75</td>
      <td>236.25</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.25</td>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>10797/11</td>
      <td>10797</td>
      <td>11</td>
      <td>21.00</td>
      <td>20</td>
      <td>0.00</td>
      <td>0</td>
      <td>21.00</td>
      <td>420.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>10800/11</td>
      <td>10800</td>
      <td>11</td>
      <td>21.00</td>
      <td>50</td>
      <td>0.10</td>
      <td>1</td>
      <td>18.90</td>
      <td>945.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.10</td>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>10823/11</td>
      <td>10823</td>
      <td>11</td>
      <td>21.00</td>
      <td>20</td>
      <td>0.10</td>
      <td>1</td>
      <td>18.90</td>
      <td>378.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.10</td>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>10842/11</td>
      <td>10842</td>
      <td>11</td>
      <td>21.00</td>
      <td>15</td>
      <td>0.00</td>
      <td>0</td>
      <td>21.00</td>
      <td>315.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>10862/11</td>
      <td>10862</td>
      <td>11</td>
      <td>21.00</td>
      <td>25</td>
      <td>0.00</td>
      <td>0</td>
      <td>21.00</td>
      <td>525.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>10869/11</td>
      <td>10869</td>
      <td>11</td>
      <td>21.00</td>
      <td>10</td>
      <td>0.00</td>
      <td>0</td>
      <td>21.00</td>
      <td>210.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30</th>
      <td>10889/11</td>
      <td>10889</td>
      <td>11</td>
      <td>21.00</td>
      <td>40</td>
      <td>0.00</td>
      <td>0</td>
      <td>21.00</td>
      <td>840.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>31</th>
      <td>10912/11</td>
      <td>10912</td>
      <td>11</td>
      <td>21.00</td>
      <td>40</td>
      <td>0.25</td>
      <td>1</td>
      <td>15.75</td>
      <td>630.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.25</td>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>32</th>
      <td>10926/11</td>
      <td>10926</td>
      <td>11</td>
      <td>21.00</td>
      <td>2</td>
      <td>0.00</td>
      <td>0</td>
      <td>21.00</td>
      <td>42.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>33</th>
      <td>10944/11</td>
      <td>10944</td>
      <td>11</td>
      <td>21.00</td>
      <td>5</td>
      <td>0.25</td>
      <td>1</td>
      <td>15.75</td>
      <td>78.75</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.25</td>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>34</th>
      <td>10986/11</td>
      <td>10986</td>
      <td>11</td>
      <td>21.00</td>
      <td>30</td>
      <td>0.00</td>
      <td>0</td>
      <td>21.00</td>
      <td>630.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35</th>
      <td>10989/11</td>
      <td>10989</td>
      <td>11</td>
      <td>21.00</td>
      <td>15</td>
      <td>0.00</td>
      <td>0</td>
      <td>21.00</td>
      <td>315.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>36</th>
      <td>11043/11</td>
      <td>11043</td>
      <td>11</td>
      <td>21.00</td>
      <td>10</td>
      <td>0.00</td>
      <td>0</td>
      <td>21.00</td>
      <td>210.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>37</th>
      <td>11073/11</td>
      <td>11073</td>
      <td>11</td>
      <td>21.00</td>
      <td>10</td>
      <td>0.00</td>
      <td>0</td>
      <td>21.00</td>
      <td>210.00</td>
      <td>11</td>
      <td>Queso Cabrales</td>
      <td>5</td>
      <td>4</td>
      <td>1 kg pkg.</td>
      <td>21.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>38</th>
      <td>10248/42</td>
      <td>10248</td>
      <td>42</td>
      <td>9.80</td>
      <td>10</td>
      <td>0.00</td>
      <td>0</td>
      <td>9.80</td>
      <td>98.00</td>
      <td>42</td>
      <td>Singaporean Hokkien Fried Mee</td>
      <td>20</td>
      <td>5</td>
      <td>32 - 1 kg pkgs.</td>
      <td>14.00</td>
      <td>0.30</td>
      <td>1</td>
      <td>0.30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>10309/42</td>
      <td>10309</td>
      <td>42</td>
      <td>11.20</td>
      <td>2</td>
      <td>0.00</td>
      <td>0</td>
      <td>11.20</td>
      <td>22.40</td>
      <td>42</td>
      <td>Singaporean Hokkien Fried Mee</td>
      <td>20</td>
      <td>5</td>
      <td>32 - 1 kg pkgs.</td>
      <td>14.00</td>
      <td>0.20</td>
      <td>1</td>
      <td>0.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>10311/42</td>
      <td>10311</td>
      <td>42</td>
      <td>11.20</td>
      <td>6</td>
      <td>0.00</td>
      <td>0</td>
      <td>11.20</td>
      <td>67.20</td>
      <td>42</td>
      <td>Singaporean Hokkien Fried Mee</td>
      <td>20</td>
      <td>5</td>
      <td>32 - 1 kg pkgs.</td>
      <td>14.00</td>
      <td>0.20</td>
      <td>1</td>
      <td>0.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>10332/42</td>
      <td>10332</td>
      <td>42</td>
      <td>11.20</td>
      <td>10</td>
      <td>0.20</td>
      <td>1</td>
      <td>8.96</td>
      <td>89.60</td>
      <td>42</td>
      <td>Singaporean Hokkien Fried Mee</td>
      <td>20</td>
      <td>5</td>
      <td>32 - 1 kg pkgs.</td>
      <td>14.00</td>
      <td>0.36</td>
      <td>1</td>
      <td>0.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>10345/42</td>
      <td>10345</td>
      <td>42</td>
      <td>11.20</td>
      <td>9</td>
      <td>0.00</td>
      <td>0</td>
      <td>11.20</td>
      <td>100.80</td>
      <td>42</td>
      <td>Singaporean Hokkien Fried Mee</td>
      <td>20</td>
      <td>5</td>
      <td>32 - 1 kg pkgs.</td>
      <td>14.00</td>
      <td>0.20</td>
      <td>1</td>
      <td>0.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>10404/42</td>
      <td>10404</td>
      <td>42</td>
      <td>11.20</td>
      <td>40</td>
      <td>0.05</td>
      <td>1</td>
      <td>10.64</td>
      <td>425.60</td>
      <td>42</td>
      <td>Singaporean Hokkien Fried Mee</td>
      <td>20</td>
      <td>5</td>
      <td>32 - 1 kg pkgs.</td>
      <td>14.00</td>
      <td>0.24</td>
      <td>1</td>
      <td>0.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>10463/42</td>
      <td>10463</td>
      <td>42</td>
      <td>11.20</td>
      <td>50</td>
      <td>0.00</td>
      <td>0</td>
      <td>11.20</td>
      <td>560.00</td>
      <td>42</td>
      <td>Singaporean Hokkien Fried Mee</td>
      <td>20</td>
      <td>5</td>
      <td>32 - 1 kg pkgs.</td>
      <td>14.00</td>
      <td>0.20</td>
      <td>1</td>
      <td>0.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>10492/42</td>
      <td>10492</td>
      <td>42</td>
      <td>11.20</td>
      <td>20</td>
      <td>0.05</td>
      <td>1</td>
      <td>10.64</td>
      <td>212.80</td>
      <td>42</td>
      <td>Singaporean Hokkien Fried Mee</td>
      <td>20</td>
      <td>5</td>
      <td>32 - 1 kg pkgs.</td>
      <td>14.00</td>
      <td>0.24</td>
      <td>1</td>
      <td>0.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>10498/42</td>
      <td>10498</td>
      <td>42</td>
      <td>14.00</td>
      <td>30</td>
      <td>0.00</td>
      <td>0</td>
      <td>14.00</td>
      <td>420.00</td>
      <td>42</td>
      <td>Singaporean Hokkien Fried Mee</td>
      <td>20</td>
      <td>5</td>
      <td>32 - 1 kg pkgs.</td>
      <td>14.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47</th>
      <td>10516/42</td>
      <td>10516</td>
      <td>42</td>
      <td>14.00</td>
      <td>20</td>
      <td>0.00</td>
      <td>0</td>
      <td>14.00</td>
      <td>280.00</td>
      <td>42</td>
      <td>Singaporean Hokkien Fried Mee</td>
      <td>20</td>
      <td>5</td>
      <td>32 - 1 kg pkgs.</td>
      <td>14.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>48</th>
      <td>10571/42</td>
      <td>10571</td>
      <td>42</td>
      <td>14.00</td>
      <td>28</td>
      <td>0.15</td>
      <td>1</td>
      <td>11.90</td>
      <td>333.20</td>
      <td>42</td>
      <td>Singaporean Hokkien Fried Mee</td>
      <td>20</td>
      <td>5</td>
      <td>32 - 1 kg pkgs.</td>
      <td>14.00</td>
      <td>0.15</td>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>49</th>
      <td>10588/42</td>
      <td>10588</td>
      <td>42</td>
      <td>14.00</td>
      <td>100</td>
      <td>0.20</td>
      <td>1</td>
      <td>11.20</td>
      <td>1,120.00</td>
      <td>42</td>
      <td>Singaporean Hokkien Fried Mee</td>
      <td>20</td>
      <td>5</td>
      <td>32 - 1 kg pkgs.</td>
      <td>14.00</td>
      <td>0.20</td>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Plot to see the distributions for:
### Group 1 - MSRP no discounts
### Group 2 - MSRP with discounts
### Group 3 - MSRP reduction and no discounts
### Group 4 - Both reduced MSRP and discounts


```python
no_discounts = order_product_df.loc[(order_product_df['net_discount'] == 0),:]
discount_only = order_product_df.loc[((order_product_df['MSRP'] == 1) & (order_product_df['Discounts'] ==1)), :]
MSRP_reduction = order_product_df.loc[((order_product_df['MSRP'] == 0) & (order_product_df['Discounts']==0)), :]
both_discounts = order_product_df.loc[((order_product_df['MSRP'] == 0) & (order_product_df['Discounts'] == 1)), :]

def four_samples_comparison(metric, alpha):
    '''The 4 sample comparison is similar to the 2-sample comparison
    
    Check first the normality of each sample
    And then calculating the ANOVA for comparison of the population mean and variance for equivalence
    
    Args:
        metric: these values for the results, such as order quantity, product revenue, customer spend, etc.
        alpha is compared to the p-value analysed by scipy or statsmodel, and defines the percentage that the result is due to chance. 
            alpha typically = 0.05,  or 5%
        
    Returns:
        group1, group2, group3, group4  for plotting. These groups are filtered based on the metric, and
        defined by the definition of discount types     
    '''
    
    group1 = no_discounts[metric]
    group2 = discount_only[metric]
    group3 = MSRP_reduction[metric]
    group4 = both_discounts[metric]
    
    print(f'The population size for group 1 is {len(group1)}, and for group 2 is {len(group2)}')
    print(f'The population size for group 3 is {len(group3)}, and for group 4 is {len(group4)}')
    
    normdist_test(group1, group2, alpha)
    normdist_test(group3, group4, alpha)
    
    fig= plt.figure(figsize=(8,4))
    plt.bar(x='MSRP, no discount', height = group1.mean(), width = 0.4, yerr = stat.sem(group1))
    plt.bar(x='MSRP with discount', height = group2.mean(), width = 0.4, yerr = stat.sem(group2))
    plt.bar(x='MSRP Reduction', height = group3.mean(), width = 0.4, yerr = stat.sem(group3))    
    plt.bar(x='MSRP Reduction and Discount', height = group4.mean(), width = 0.4, yerr = stat.sem(group4))

    plt.title(f'Comparison of Impact of Discount Types on {metric}')
    plt.ylabel(f'Average {metric}');
    
    return group1, group2, group3, group4
```


```python
no_disc, other_disc, MSRP_disc, all_discs = four_samples_comparison('Quantity', alpha)
```

    The population size for group 1 is 915, and for group 2 is 582
    The population size for group 3 is 402, and for group 4 is 256
    


![png](output_69_1.png)


    Sample 1 P-value 9.269039017041992e-89 is less than alpha 0.05
    Sample 1 Normality: We can reject the null hypothesis that the 1st sample comes from a normal distribution 
    
    


![png](output_69_3.png)


    Sample 2 P-value 9.269039017041992e-89 is less than alpha 0.05
    Sample 2 Normality: We can reject the null hypothesis that the 2nd sample comes from a normal distribution 
    
    


![png](output_69_5.png)


    Sample 1 P-value 7.774633648183358e-25 is less than alpha 0.05
    Sample 1 Normality: We can reject the null hypothesis that the 1st sample comes from a normal distribution 
    
    


![png](output_69_7.png)


    Sample 2 P-value 7.774633648183358e-25 is less than alpha 0.05
    Sample 2 Normality: We can reject the null hypothesis that the 2nd sample comes from a normal distribution 
    
    


![png](output_69_9.png)



```python
# Use ANOVA to check the similarity of the populations

fstat, pval = stat.f_oneway(no_disc, other_disc, MSRP_disc, all_discs)
print(fstat, pval)
if pval < alpha:
    print(f"The null hypothesis is rejected that there is no difference between the groups and the means are equal")
    print("The populations and means are not equivalent")
else:
    print(f"The null hypothesis is accepted that there is no difference between the groups, and the means are equal")
```

    15.50787532818146 5.56608507477635e-10
    The null hypothesis is rejected that there is no difference between the groups and the means are equal
    The populations and means are not equivalent
    


```python
rev1, rev2, rev3, rev4 = four_samples_comparison('Total Product Spend ($)', alpha)
```

    The population size for group 1 is 915, and for group 2 is 582
    The population size for group 3 is 402, and for group 4 is 256
    


![png](output_71_1.png)


    Sample 1 P-value 7.972762931381562e-264 is less than alpha 0.05
    Sample 1 Normality: We can reject the null hypothesis that the 1st sample comes from a normal distribution 
    
    


![png](output_71_3.png)


    Sample 2 P-value 7.972762931381562e-264 is less than alpha 0.05
    Sample 2 Normality: We can reject the null hypothesis that the 2nd sample comes from a normal distribution 
    
    


![png](output_71_5.png)


    Sample 1 P-value 1.2126946852158827e-129 is less than alpha 0.05
    Sample 1 Normality: We can reject the null hypothesis that the 1st sample comes from a normal distribution 
    
    


![png](output_71_7.png)


    Sample 2 P-value 1.2126946852158827e-129 is less than alpha 0.05
    Sample 2 Normality: We can reject the null hypothesis that the 2nd sample comes from a normal distribution 
    
    


![png](output_71_9.png)



```python
fstat, pval = stat.f_oneway(rev1, rev2, rev3, rev4)
print(fstat, pval)
if pval < alpha:
    print(f"The null hypothesis is rejected that there is no difference between the groups and the means are equal")
else:
    print(f"The null hypothesis is accepted that there is no difference between the groups, and the means are equal")
```

    1.9497649299486008 0.11951841800067911
    The null hypothesis is accepted that there is no difference between the groups, and the means are equal
    

## Result - there is a significant difference between pricing strategies.
## MSRP reduction is the least effective in generating more orders.

## Pricing through discounts is a more effective strategies to get more orders

## However, discounting has no significant impact on improving product sales.  There was no significant difference between no discounts, and other discounts.  Interestingly,  MSRP reduction had a significant level of lower revenue, which is undesirable

## The results can be interpreted through understanding the nature of competition for consumer goods.   Competition is usually high -- the ability to differentiate consumer products through unique features or quality is limited.  Thus, it is typical to rely on differentiation based on price.
    * As shown in this analysis, lowering MSRP doesn't make a lot of sense.  Neither Order Quantity nor Product Sales are improved through lowering MSRP.
    
    * This analysis shows it was a good idea to move away from simply lowering the MSRP, which is a baseline price, everyday price in the mind of the consumer.   Instead a more effective way to generate more orders is to position as "temporary price reductions" which the analysis shows can generate more orders,  while maintaining the Product Revenue,  which may be necessary in the face of competition.


#  Hypothesis 4:  Confirmation that MSRP Reduction strategy impact on Order Quantity and Product Spend


```python
MSRP_reduction, no_MSRP_reduction = two_sample_tests(order_product_df, "MSRP", 'Quantity')
plot_two_sample_tests(no_MSRP_reduction, MSRP_reduction, 'MSRP Discount', 'Quantity')

```

    Number of datapoints in group 1, no MSRP = 658
    Number of datapoints in group 2, with MSRP = 1497
    Number of datapoints in the samples > 30, possible to assume normality through the Central Limit Theorem 
    
    Check the normality anyway!
    


![png](output_75_1.png)


    Sample 1 P-value 4.3519853073007804e-44 is less than alpha 0.05
    Sample 1 Normality: We can reject the null hypothesis that the 1st sample comes from a normal distribution 
    
    


![png](output_75_3.png)


    Sample 2 P-value 4.3519853073007804e-44 is less than alpha 0.05
    Sample 2 Normality: We can reject the null hypothesis that the 2nd sample comes from a normal distribution 
    
    Levene test: We accept the null hypothesis that the variances of the populations equal 
    
    Need to use the non-parametric test (Mann-WhitneyU) to determine the equivalence of the populations 
    
    MWU: The U statistic is 465219.5, and the p-value is 0.01990546853117364
    MWU: The MannWhitneyU p-value 0.01990546853117364 is less than alpha 0.05
    MannWhitneyU test: We can reject the null hypothesis that the two populations are equal 
    
    The average Quantity sold, without MSRP Discount is 23.49632598530394
    The average Quantity sold, with MSRP Discount is 24.533434650455927
    


![png](output_75_5.png)



```python
MSRP_reduction, no_MSRP_reduction = two_sample_tests(order_product_df, "MSRP", 'Total Product Spend ($)')
plot_two_sample_tests(no_MSRP_reduction, MSRP_reduction, 'MSRP Discount', 'Total Product Spend ($)')
```

    Number of datapoints in group 1, no MSRP = 658
    Number of datapoints in group 2, with MSRP = 1497
    Number of datapoints in the samples > 30, possible to assume normality through the Central Limit Theorem 
    
    Check the normality anyway!
    


![png](output_76_1.png)


    Sample 1 P-value 1.3828420073540596e-179 is less than alpha 0.05
    Sample 1 Normality: We can reject the null hypothesis that the 1st sample comes from a normal distribution 
    
    


![png](output_76_3.png)


    Sample 2 P-value 1.3828420073540596e-179 is less than alpha 0.05
    Sample 2 Normality: We can reject the null hypothesis that the 2nd sample comes from a normal distribution 
    
    Levene test: We accept the null hypothesis that the variances of the populations equal 
    
    Need to use the non-parametric test (Mann-WhitneyU) to determine the equivalence of the populations 
    
    MWU: The U statistic is 463835.0, and the p-value is 0.015554193286930031
    MWU: The MannWhitneyU p-value 0.015554193286930031 is less than alpha 0.05
    MannWhitneyU test: We can reject the null hypothesis that the two populations are equal 
    
    The average Total Product Spend ($) sold, without MSRP Discount is 611.4722408149636
    The average Total Product Spend ($) sold, with MSRP Discount is 532.5518161094229
    


![png](output_76_5.png)


# Hypothesis 5: Look at net discounts impact on order quantity and total product spend


```python
# Run the two_sample_tests to see if there is an Net Discount impact on Order Quantity or Total Product Spend ($)
no_discount, with_net_discount = two_sample_tests(order_product_df, 'net_discount', 'Quantity')

plot_two_sample_tests(no_discount, with_net_discount, 'Net Discounts', 'Quantity')
```

    Number of datapoints in group 1, no net_discount = 915
    Number of datapoints in group 2, with net_discount = 1240
    Number of datapoints in the samples > 30, possible to assume normality through the Central Limit Theorem 
    
    Check the normality anyway!
    


![png](output_78_1.png)


    Sample 1 P-value 9.269039017041992e-89 is less than alpha 0.05
    Sample 1 Normality: We can reject the null hypothesis that the 1st sample comes from a normal distribution 
    
    


![png](output_78_3.png)


    Sample 2 P-value 9.269039017041992e-89 is less than alpha 0.05
    Sample 2 Normality: We can reject the null hypothesis that the 2nd sample comes from a normal distribution 
    
    Levene test: We accept the null hypothesis that the variances of the populations equal 
    
    Need to use the non-parametric test (Mann-WhitneyU) to determine the equivalence of the populations 
    
    MWU: The U statistic is 494688.0, and the p-value is 1.7358140845437264e-07
    MWU: The MannWhitneyU p-value 1.7358140845437264e-07 is less than alpha 0.05
    MannWhitneyU test: We can reject the null hypothesis that the two populations are equal 
    
    The average Quantity sold, without Net Discounts is 21.778142076502732
    The average Quantity sold, with Net Discounts is 25.31451612903226
    


![png](output_78_5.png)


## With Net discounts the Total Product Spend is statistically equal to the Total product spend (revenue) without discounts.

## In other words,  it doesn't really pay off to offer discounts.

## However, this analysis does not reflect the entire picture. In the face of competitors who may offer discounts,  Northwind may have to offer discounts, simply to keep up the appearances with competitors, in order to get orders in the first place, without any increase in revenue


```python
#Check the effect of Net Discounts on Total Product Spend
no_discount, with_net_discount = two_sample_tests(order_product_df, 'net_discount', 'Total Product Spend ($)')

plot_two_sample_tests(no_discount, with_net_discount, 'Net Discounts', 'Total Product Spend ($)')
```

    Number of datapoints in group 1, no net_discount = 915
    Number of datapoints in group 2, with net_discount = 1240
    Number of datapoints in the samples > 30, possible to assume normality through the Central Limit Theorem 
    
    Check the normality anyway!
    


![png](output_80_1.png)


    Sample 1 P-value 7.972762931381562e-264 is less than alpha 0.05
    Sample 1 Normality: We can reject the null hypothesis that the 1st sample comes from a normal distribution 
    
    


![png](output_80_3.png)


    Sample 2 P-value 7.972762931381562e-264 is less than alpha 0.05
    Sample 2 Normality: We can reject the null hypothesis that the 2nd sample comes from a normal distribution 
    
    Levene test: We accept the null hypothesis that the variances of the populations equal 
    
    Need to use the non-parametric test (Mann-WhitneyU) to determine the equivalence of the populations 
    
    MWU: The U statistic is 564123.5, and the p-value is 0.41198294193148727
    MannWhitneyU test: We can accept the null hypothesis that the two populations are equal 
    
    The average Total Product Spend ($) sold, without Net Discounts is 608.5897377049181
    The average Total Product Spend ($) sold, with Net Discounts is 571.7205076612905
    


![png](output_80_5.png)



```python
'''Tukey tests show that none of the different discounts has a meaningful impact on the Total Product Spend
data = orderproduct_df['Quantity'].values 
labels = orderproduct_df['Net Discounts'].values

print(data), print(labels)

import statsmodels.api as sms

model = sms.stats.multicomp.pairwise_tukeyhsd(data,labels) 
model.summary()'''
```


```python
'''model.plot_simultaneous()
plt.ylabel('Net Discounts')
plt.xlabel('Quantity');'''
```


```python

```

# Northwind Sales Background information

The Deliverables
The goal of your project is to query the database to get the data needed to perform a statistical analysis. In this statistical analysis, you'll need to perform a hypothesis test (or perhaps several) to answer the following question:

##### Does discount amount have a statistically significant effect on the quantity of a product in an order? If so, at what level(s) of discount?

In addition to answering this question with a hypothesis test, you will also need to come up with at least 3 other hypotheses to test on your own. These can by anything that you think could be imporant information for the company.

For this hypothesis, be sure to specify both the null hypothesis and the alternative hypothesis for your question. You should also specify if this is one-tail or a two-tail test.

For online students, there will be four deliverables for this project:

1) A Jupyter Notebook containing any code you've written for this project. This work will need to be pushed to your GitHub repository in order to submit your project.

2) An organized README.md file in the GitHub repository that describes the contents of the repository. This file should be the source of information for navigating through the repository.

3) A Blog Post.

4) An "Executive Summary" PowerPoint Presentation that explains the hypothesis tests you ran, your findings, and their relevance to company stakeholders.

Jupyter Notebook Must-Haves
For this project, your Jupyter Notebook should meet the following specifications

Organization/Code Cleanliness

The notebook should be well organized, easy to follow, and code is commented where appropriate.

* Level Up: The notebook contains well-formatted, professional looking markdown cells explaining any substantial code. All functions have docstrings that act as professional-quality documentation.

* The notebook is written to technical audiences with a way to both understand your approach and reproduce your results. The target audience for this deliverable is other data scientists looking to validate your findings.

* Any SQL code written to source data should also be included.

Findings

Your notebook should clearly show how you arrived at your results for each hypothesis test, including how you calculated your p-values.

* You should also include any other statistics that you find relevant to your analysis, such as effect size.
