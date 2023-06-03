#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings

from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")


# # Loading the dataset

# In[2]:


train_df=pd.read_csv(r"C:\Users\HP\Downloads\tcs projec dataset\train_data.csv")


# In[3]:


test_df=pd.read_csv(r"C:\Users\HP\Downloads\tcs projec dataset\test_data.csv")


# In[4]:


price_df=pd.read_csv(r"C:\Users\HP\Downloads\tcs projec dataset\product_prices.csv")


# In[5]:


date_df=pd.read_csv(r"C:\Users\HP\Downloads\tcs projec dataset\date_to_week_id_map.csv")


# In[6]:


sample_df=pd.read_csv(r"C:\Users\HP\Downloads\tcs projec dataset\sample_submission.csv")


# In[7]:


train_df


# In[8]:


test_df


# In[9]:


price_df


# In[10]:


date_df


# In[11]:


sample_df


# ## Joining the datasets with common column

# In[12]:


train_df.columns


# In[13]:


#joining price and date datasets using inner join

df=pd.merge(price_df,date_df, on=['week_id'], how='inner')


# In[14]:


df1=pd.merge(train_df,df, on=['date','product_identifier','outlet'], how='inner')
df1.head()


# # ----------------------------------------------------------------------------------

# # Data Analysis

# In[15]:


df1.shape


# In[16]:


df1.info()


# In[17]:


df1.describe()


# In[18]:


# Checking for Null Values
df1.isna().sum()


# In[19]:


df1.columns


# In[20]:


#checking the cloumns having Dtype 'O'


# In[21]:


df1['category_of_product'].unique()


# In[22]:


df1['state'].unique()


# In[23]:


#converting the date column data type to data and time


# In[24]:


data=df1.copy()


# In[25]:


data1=df1.copy()


# In[26]:


data['date'] = pd.to_datetime(data['date'])


# In[27]:


data.info()


# In[28]:


# Setting index to 'datetime'
data.set_index('date', inplace=True)


# In[29]:


data


# # --------------------------------------------------------------------------------------------

# # Feature Engineering

# In[30]:


data=data.reset_index('date')


# In[31]:


# Extracting additional features from 'date'
data['year'] = pd.to_datetime(data['date']).dt.year
data['month'] = pd.to_datetime(data['date']).dt.month
data['day'] = pd.to_datetime(data['date']).dt.day


# In[32]:


data=data.set_index('date')


# In[33]:


data


# In[34]:


# Grouping feature into categorical data and numerical data


# In[35]:


data.info()


# In[36]:


categorical_features=[features for features in data.columns if data[features].dtypes=="O"]


# In[37]:


numerical_features=[features for features in data.columns if data[features].dtypes!="O" and features not in ['date','year', 'month','day','week_id']]
numerical_features


# In[38]:


#Seperating numerical features into contionus and discrete features.
continous_features=[features for features in numerical_features if len(data[features].unique()) > 50 ]


# In[39]:


discrete_features =[features for features in numerical_features if len(data[features].unique()) <= 50]


# In[40]:


categorical_features


# In[41]:


continous_features


# In[42]:


discrete_features


# # ----------------------------------------------------------------------------

# ### Splitting Dataset based on Category of Product

# In[43]:


data['category_of_product'].unique()


# ## Drinks and Food

# In[44]:


#Now we create a dataframe which includes only data related to drinks_and_food sales. 
drinks_and_food= data.loc[data['category_of_product'] == 'drinks_and_food']
drinks_and_food


# In[45]:


#Now let's sort drinks_and_food dataframe according to date
drinks_and_food = drinks_and_food.sort_values('date')
drinks_and_food


# In[46]:


#Now let's find how much total drinks_and_food sales occurred on each date
drinks_and_food1 =drinks_and_food.groupby('date')['sales'].sum().reset_index('date')


# In[47]:


drinks_and_food1


# In[48]:


#Now let's set the date column as the index column
drinks_and_food1 = drinks_and_food1.set_index('date')
drinks_and_food1


# In[49]:


#Now let's see whether there is any frequency in the dataframe
drinks_and_food1.index


# In[50]:


#Now let's resample the data into means of monthly sales of drinks_and_food and save this into a new variable called date
y_drinks_and_food = drinks_and_food1['sales'].resample('MS').mean()
y_drinks_and_food


# In[51]:


#Now let's check the monthly sales value happened in year 2013
print(y_drinks_and_food['2013':])


# ## Fast Moving Consumer Goods

# In[52]:


#Now we create a dataframe which includes only data related to drinks_and_food sales. 
fast_moving_consumer_goods= data.loc[data['category_of_product'] == 'fast_moving_consumer_goods']
fast_moving_consumer_goods


# In[53]:


#Now let's sort fast_moving_consumer_goods dataframe according to date
fast_moving_consumer_goods= fast_moving_consumer_goods.sort_values('date')
fast_moving_consumer_goods


# In[54]:


#Now let's find how much total fast_moving_consumer_goods sales occurred on each date
fast_moving_consumer_goods1 =fast_moving_consumer_goods.groupby('date')['sales'].sum().reset_index()


# In[55]:


fast_moving_consumer_goods1


# In[56]:


#Now let's set the date column as the index column
fast_moving_consumer_goods1 = fast_moving_consumer_goods1.set_index('date')
fast_moving_consumer_goods1


# In[57]:


#Now let's see whether there is any frequency in the dataframe
fast_moving_consumer_goods1.index


# In[58]:


#Now let's resample the data into means of monthly sales of fast_moving_consumer_goods and save this into a new variable called date
y_fast_moving_consumer_goods = fast_moving_consumer_goods1['sales'].resample('MS').mean()
y_fast_moving_consumer_goods


# In[59]:


#Now let's check the monthly sales value happened in year 2013
print(y_fast_moving_consumer_goods['2013':])


# ## Others

# In[60]:


#Now we create a dataframe which includes only data related to others sales. 
others= data.loc[data['category_of_product'] == 'others']
others


# In[61]:


#Now let's sort others dataframe according to date
others = others.sort_values('date')
others


# In[62]:


#Now let's find how much total others sales occurred on each date
others1 =others.groupby('date')['sales'].sum().reset_index()


# In[63]:


others1


# In[64]:


#Now let's set the date column as the index column
others1 = others1.set_index('date')
others1


# In[65]:


#Now let's see whether there is any frequency in the dataframe
others1.index


# In[66]:


#Now let's resample the data into means of monthly sales of others and save this into a new variable called date
y_others = others1['sales'].resample('MS').mean()
y_others


# In[67]:


#Now let's check the monthly sales value happened in year 2013
print(y_others['2013':])


# ## Arranging Data Product-wise

# In[68]:


sales_grouped = data.groupby(['product_identifier', 'date']).sum()

#sales_grouped

# Group the sales by product identifier and date
sales_grouped = pd.pivot_table(data, values='sales', index='date', columns='product_identifier', aggfunc=sum)


#sales_grouped

sales_monthly = sales_grouped.resample('MS').mean()

sales_monthly

#sales_monthly.info()


# In[69]:


price_grouped = data.groupby(['product_identifier', 'date']).sum()

#sales_grouped

# Group the sales by product identifier and date
price_grouped = pd.pivot_table(data, values='sell_price', index='date', columns='product_identifier', aggfunc=sum)


#sales_grouped

price_monthly = price_grouped.resample('MS').mean()

price_monthly

#sales_monthly.info()


# # ----------------------------------------------------------------------------------------

# # DATA VISUALIZATION

# ### Plotting the sales data for each categories

# In[70]:


drinks_and_food1.plot(figsize=(20,5))


# In[71]:


fast_moving_consumer_goods1.plot(figsize=(20,5))


# In[72]:


others1.plot(figsize=(20,5))


# # Plotting the mean sales data for each categories

# In[73]:


y_drinks_and_food.plot(figsize=(15,6))
plt.title("drinks_and_food supplies sales")
plt.ylabel("Sales")
plt.show()


# In[74]:


y_fast_moving_consumer_goods.plot(figsize=(15,6))
plt.title("fast_moving_consumer_goods supplies sales")
plt.ylabel("Sales")
plt.show()


# In[75]:


y_others.plot(figsize=(15,6))
plt.title("others supplies sales")
plt.ylabel("Sales")
plt.show()


# # ------------------------------------------------------------------------------------

# # Performing ETS Decomposition

# # a) Drinks and Food

# In[76]:


from pylab import rcParams
rcParams['figure.figsize']=18,8
decomposition_drinks_and_food = sm.tsa.seasonal_decompose(y_drinks_and_food,model='additive')
fig = decomposition_drinks_and_food.plot()
plt.show()


# In[77]:


decomposition_drinks_and_food.trend.plot(figsize=(18,5))


# In[78]:


decomposition_drinks_and_food.seasonal.plot(figsize=(18,5))


# # b) Fast MovingConsumer Goods

# In[79]:


from pylab import rcParams
rcParams['figure.figsize']=18,8
decomposition_fast_moving_consumer_goods = sm.tsa.seasonal_decompose(y_fast_moving_consumer_goods,model='additive')
fig = decomposition_fast_moving_consumer_goods.plot()
plt.show()


# In[80]:


decomposition_fast_moving_consumer_goods.trend.plot(figsize=(18,5))


# In[81]:


decomposition_fast_moving_consumer_goods.seasonal.plot(figsize=(18,5))


# ## c) Others

# In[82]:


from pylab import rcParams
rcParams['figure.figsize']=18,8
decomposition_others = sm.tsa.seasonal_decompose(y_others,model='additive')
fig = decomposition_others.plot()
plt.show()


# In[83]:


decomposition_others.trend.plot(figsize=(18,5))


# In[84]:


decomposition_others.seasonal.plot(figsize=(18,5))


# In[85]:


import itertools
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

for col in sales_monthly.columns:
    y = sales_monthly.loc[:, [col]]
    var_index = y.columns[0]  # Get the variable index

    # Perform time series decomposition
    result = seasonal_decompose(y, model='additive', period=12)

    # Plot the decomposition
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    fig.suptitle('Time series decomposition of Average Sales for product ID {}'.format(var_index))
    ax1.set_ylabel('Observed')
    ax1.plot(y.index, y)
    ax2.set_ylabel('Trend')
    ax2.plot(y.index, result.trend)
    ax3.set_ylabel('Seasonal')
    ax3.plot(y.index, result.seasonal)
    ax4.set_ylabel('Residual')
    ax4.plot(y.index, result.resid)
    plt.show()


# NOTE:Every product and category has seasonality

# # ---------------------------------------------------------------------------

# # Data visualization -part2

# In[86]:


sales_per_month=data['sales'].resample('M').sum()
sales_per_day=data['sales'].resample('D').sum()


# In[87]:


# Plotting total sales per month

plt.figure(figsize=(15,7))
plt.title("Sales per month")
sns.lineplot(data = sales_per_month, dashes=False)
plt.show()


# In[88]:


# Plotting total sales per day

plt.figure(figsize=(15,7))
plt.title("Sales per day")
sns.lineplot(data = sales_per_day, dashes=False)
plt.show()


# In[89]:


# Plotting the Distribution of total sales data

plt.figure(figsize=(8,6))
sns.distplot(sales_per_month)
plt.title('Total Sales Distribution')
plt.show()


# In[90]:


# Plotting the Distribution of total sales per day data

plt.figure(figsize=(8,6))
sns.distplot(sales_per_day)
plt.title('Total Sales per day Distribution')
plt.show()


# In[91]:


data=data.reset_index()


# In[92]:


data['date'] = pd.to_datetime(data['date'])
# Setting index to 'datetime'
data.set_index('date', inplace=True)


# In[93]:


category_data = data[data['category_of_product'] == 'drinks_and_food']

# Group data by month and calculate the average sell price for each month
monthly_avg_sell_price = category_data.resample('M')['sell_price'].mean()

# Plot the results
plt.figure(figsize=(12,6))
sns.lineplot(x=monthly_avg_sell_price.index, y=monthly_avg_sell_price)
plt.title('Average selling price of drinks and food')
plt.xlabel('Month')
plt.ylabel('Average sell price')
plt.show()


# In[94]:


data = data.reset_index()


# In[95]:


# Group the data by month and calculate the mean selling price for each month
monthly_data = data.groupby(pd.Grouper(key='date', freq='M')).mean()

# Create a line plot of the average selling price per month
plt.plot(monthly_data.index, monthly_data['sell_price'])

# Set the x-axis label
plt.xlabel('Month')

# Set the y-axis label
plt.ylabel('Average Selling Price')

# Set the title
plt.title('Average Selling Price per Month')

# Show the plot
plt.show()


# In[96]:


total_sales = data.groupby('state')['sales'].sum().reset_index()

# Sort values by sales in descending order
total_sales = total_sales.sort_values(by='sales', ascending=False)

colors = ['orange','purple', 'green']



# Create vertical bar plot
fig, ax = plt.subplots(figsize=(8,6))
ax.bar(total_sales['state'], total_sales['sales'],width=0.8, color=colors)

# Set labels and title
ax.grid(False)
ax.set_xlabel('State')
ax.set_ylabel('Total Sales')
ax.set_title('Total Sales per State')

plt.show()


# In[97]:


total_sales = data.groupby('category_of_product')['sales'].sum().reset_index()

# Sort values by sales in descending order
total_sales = total_sales.sort_values(by='sales', ascending=False)

colors = ['black','blue', 'brown']



# Create vertical bar plot
fig, ax = plt.subplots(figsize=(8,6))
ax.bar(total_sales['category_of_product'], total_sales['sales'],width=0.8, color=colors)

# Set labels and title
ax.grid(False)
ax.set_xlabel('Category of product')
ax.set_ylabel('Total Sales')
ax.set_title('Total Sales per category')

plt.show()


# In[98]:


average_price = data.groupby('category_of_product')['sell_price'].mean().reset_index()

# Sort values by selling price in descending order
average_price = average_price.sort_values(by='sell_price', ascending=False)

colors = ['blue', 'brown','black']



# Create vertical bar plot
fig, ax = plt.subplots(figsize=(8,6))
ax.bar(average_price['category_of_product'], average_price['sell_price'],width=0.8, color=colors)

# Set labels and title
ax.grid(False)
ax.set_xlabel('Category of product')
ax.set_ylabel('Average_selling_price')
ax.set_title('Average selling price per category')

plt.show()


# In[99]:


plt.figure(figsize=(8,6))
sns.barplot(x=data['department_identifier'], y=data['sales'])
plt.title('Sales by different Departments')
plt.show()


# In[100]:


# Plotting average sales with respect to outlets in a day

plt.figure(figsize=(12,6))
sns.barplot(x=data['outlet'], y=data['sales'])
plt.title('average sales at different outlets in a day')
plt.show()


# In[101]:


# Plotting average selling price with respect to departments 

plt.figure(figsize=(12,6))
sns.barplot(x=data['department_identifier'], y=data['sell_price'])
plt.title('average selling price of different department ')
plt.show()


# In[102]:


plt.figure(figsize=(12,6))
sns.barplot(x=data['product_identifier'], y=data['sell_price'])
plt.title('Selling price of different products')
plt.xticks(rotation=90)
plt.show()


# In[103]:


# Plotting average selling price with respect to products

plt.figure(figsize=(20,10))
sns.barplot(x=data['product_identifier'], y=data['sell_price'],hue=data['category_of_product'])
plt.title('average selling price of different products')
plt.xticks(rotation=45)
plt.show()


# In[104]:


# Plotting average sales with respect to products in a day

plt.figure(figsize=(20,15))
sns.barplot(x=data['product_identifier'], y=data['sales'],hue=data['category_of_product'])
plt.title('average sales of products in a day')
plt.xticks(rotation=45,ha='right')
plt.show()


# In[105]:


price_per_month = data.groupby(['year', 'month'])['sell_price'].mean().reset_index()

# Filter the data for the two years you want to compare
year1_price = price_per_month[price_per_month['year'] == 2012]
year2_price = price_per_month[price_per_month['year'] == 2013]
year3_price = price_per_month[price_per_month['year'] == 2014]

# Plot a line graph of the sales for each month in the two years
plt.plot(year1_price['month'], year1_price['sell_price'], label='2012')
plt.plot(year2_price['month'], year2_price['sell_price'], label='2013')
plt.plot(year3_price['month'], year3_price['sell_price'], label='2014')
plt.xlabel('Month')
plt.ylabel('Average Selling Price')
plt.title('Price Comparison between years 2012,2013 and 2014')
plt.legend()
plt.show()


# In[106]:


import seaborn as sns
import matplotlib.pyplot as plt
data5=data.copy()
data5=data5.reset_index()
# Filter the data for the desired category
category_data = data5[data5['category_of_product'] == 'drinks_and_food']

# Extract the year and month from the 'date' column
category_data['year'] = category_data['date'].dt.year
category_data['month'] = category_data['date'].dt.month

# Group the data by year and month, and calculate the average selling price
avg_price = category_data.groupby(['year', 'month'])['sell_price'].mean().reset_index()

# Plot the average selling price per month for each year using a line plot
plt.figure(figsize=(12, 6))
sns.lineplot(x='month', y='sell_price', hue='year', data=avg_price)
plt.title('Average Selling Price per Month for ' + category_data['category_of_product'].iloc[0])
plt.xlabel('Month')
plt.ylabel('Average Selling Price')
plt.show()


# In[107]:


import seaborn as sns
import matplotlib.pyplot as plt
data5=data.copy()
data5=data5.reset_index()
# Filter the data for the desired category
category_data = data5[data5['category_of_product'] == 'fast_moving_consumer_goods']

# Extract the year and month from the 'date' column
category_data['year'] = category_data['date'].dt.year
category_data['month'] = category_data['date'].dt.month

# Group the data by year and month, and calculate the average selling price
avg_price = category_data.groupby(['year', 'month'])['sell_price'].mean().reset_index()

# Plot the average selling price per month for each year using a line plot
plt.figure(figsize=(12, 6))
sns.lineplot(x='month', y='sell_price', hue='year', data=avg_price)
plt.title('Average Selling Price per Month for ' + category_data['category_of_product'].iloc[0])
plt.xlabel('Month')
plt.ylabel('Average Selling Price')
plt.show()


# In[108]:


import seaborn as sns
import matplotlib.pyplot as plt
data5=data.copy()
data5=data5.reset_index()
# Filter the data for the desired category
category_data = data5[data5['category_of_product'] == 'others']

# Extract the year and month from the 'date' column
category_data['year'] = category_data['date'].dt.year
category_data['month'] = category_data['date'].dt.month

# Group the data by year and month, and calculate the average selling price
avg_price = category_data.groupby(['year', 'month'])['sell_price'].mean().reset_index()

# Plot the average selling price per month for each year using a line plot
plt.figure(figsize=(12, 6))
sns.lineplot(x='month', y='sell_price', hue='year', data=avg_price)
plt.title('Average Selling Price per Month for ' + category_data['category_of_product'].iloc[0])
plt.xlabel('Month')
plt.ylabel('Average Selling Price')
plt.show()


# In[109]:


data['product_identifier'].unique()


# In[110]:


prod_id=[74,  337,  423,  432,  581,  611,  631,  659,  743,  797,  868,
        904,  926,  972,  973, 1054, 1135, 1173, 1190, 1196, 1228, 1240,
       1242, 1275, 1322, 1328, 1365, 1424, 1472, 1508, 1542, 1548, 1599,
       1629, 1672, 1694, 1727, 1753, 2294, 2332, 2492, 2768, 2794, 2818,
       2853, 2932, 2935, 3004, 3008, 3021]
for i in prod_id:
    product_data = data5[data5['product_identifier']==i]

# Convert date column to datetime format
    product_data['date'] = pd.to_datetime(product_data['date'])

# Create a column for year and month
    product_data['year_month'] = product_data['date'].dt.strftime('%Y-%m')

# Calculate average selling price per month for each year
    avg_price = product_data.groupby([product_data['date'].dt.year, product_data['date'].dt.month]).mean()['sell_price']
    avg_price = avg_price.unstack(level=0)

# Plot the comparison using line chart
    avg_price.plot(kind='line', figsize=(6,5), marker='o')

# Set title and labels
    plt.title('Comparison of Average Selling Price of Product {} by Month'.format(i))
    plt.xlabel('Month')
    plt.ylabel('Average Selling Price')

# Show plot
    plt.show()


# In[111]:


product_data = data5[data5['product_identifier']==2935] #put 50 products


# Convert date column to datetime format
product_data['date'] = pd.to_datetime(product_data['date'])

# Create a column for year and month
product_data['year_month'] = product_data['date'].dt.strftime('%Y-%m')

# Calculate average selling price per month for each year
avg_price = product_data.groupby([product_data['date'].dt.year, product_data['date'].dt.month]).mean()['sell_price']
avg_price = avg_price.unstack(level=0)

# Plot the comparison using line chart
avg_price.plot(kind='line', figsize=(12,6), marker='o')

# Set title and labels
plt.title('Comparison of Average Selling Price of Product 2935 by Month')
plt.xlabel('Month')
plt.ylabel('Average Selling Price')

# Show plot
plt.show()


# In[112]:


# Calculate total sales by category
sales_by_category = data.groupby('category_of_product')['sales'].sum().reset_index()

# Create a pie chart
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(sales_by_category['sales'], labels=sales_by_category['category_of_product'], autopct='%1.1f%%', startangle=90)

# Set title
ax.set_title('Sales by Category of Product')

plt.show()


# In[113]:


# Calculate total sales per department
total_sales = data.groupby('department_identifier')['sales'].sum().reset_index()

# Create a pie chart
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(total_sales['sales'], labels=total_sales['department_identifier'], autopct='%1.1f%%', startangle=90)
plt.setp(ax.texts, rotation=330, va='center', ha='left')


# Set title
ax.set_title('Sales per Department')

plt.show()


# In[114]:


import seaborn as sns

# Group data by product identifier and calculate mean selling price and total sales
grouped_data = data.groupby('product_identifier').agg({'sell_price': 'mean', 'sales': 'sum'}).reset_index()

# Create a regplot
sns.regplot(x=grouped_data['sales'], y=grouped_data['sell_price'])

# Set labels and title
plt.xlabel('Total Sales')
plt.ylabel('Mean Selling Price')
plt.title('Relationship between Total Sales and Mean Selling Price')

plt.show()#good fit


# In[115]:


# Calculate mean selling price per product
mean_sell_price = data.groupby('product_identifier')['sell_price'].mean().reset_index()

# Calculate total sales per product
total_sales = data.groupby('product_identifier')['sales'].sum().reset_index()

# Merge dataframes on product identifier
merged_df = pd.merge(mean_sell_price, total_sales, on='product_identifier')

# Create scatterplot
plt.figure(figsize=(8, 6))
plt.scatter( merged_df['sales'],merged_df['sell_price'], alpha=0.5)

# Set labels and title
plt.ylabel('Mean Selling Price')
plt.xlabel('Total Sales')
plt.title('Total Sales vs. Mean Selling Price')

plt.show()


# In[116]:


sns.pairplot(data[['product_identifier', 'department_identifier', 'category_of_product', 'outlet', 'state', 'sales', 'sell_price']])


# # -------------------------------------------------------------------------------

# # Correlation

# In[117]:


data10=data.drop(['day','week_id','year','month'],axis=1)
corrmatrix = data10.corr()
plt.subplots(figsize=(20,10))
sns.heatmap(corrmatrix,annot=True,cmap = 'YlGnBu')


# # Outlier Detection

# In[118]:


for feature in continous_features:
    data.boxplot(column= feature )
    plt.xlabel(feature)
    plt.title(feature)
    plt.show()


# In[119]:


sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10,8))

sns.boxplot(x="category_of_product", y="sales", data=data, ax=ax)

ax.set_title("Boxplot of Category of Product vs Sales")
ax.set_xlabel("Category of Product")
ax.set_ylabel("Sales")

plt.show()


# In[120]:


def find_outliers_IQR(data):
    q1=data.quantile(0.25)
    q3=data.quantile(0.75)
    IQR=q3-q1
    outliers = data[((data<(q1-1.5*IQR)) | (data>(q3+1.5*IQR)))]
    
    return outliers

for feature in continous_features:
    outliers = find_outliers_IQR(data[feature])

    print(feature)
    print('number of outliers: '+ str(len(outliers)))
    print('max outlier value: '+ str(outliers.max()))
    print('min outlier value: '+ str(outliers.min()))
    print('% of outliers: '+ str(len(outliers)/(len(data[feature]))*100))
    print('\n')



# In[121]:


def find_outliers_IQR(x):
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    IQR = q3 - q1
    outliers = x[((x< (q1 - 1.5 * IQR)) | (x > (q3 + 1.5 * IQR)))]

    return outliers

# Set the product identifier for which you want to find outliers
product_ids = [  74,  337,  423,  432,  581,  611,  631,  659,  743,  797,  868,904,  926,  972,  973, 1054, 1135, 1173, 1190, 1196, 1228, 1240,1242, 1275, 1322,
               1328, 1365, 1424, 1472, 1508, 1542, 1548, 1599,1629, 1672, 1694, 1727, 1753, 2294, 2332, 2492, 2768, 2794, 2818,2853, 2932, 2935, 3004, 3008, 3021]

for product_id in product_ids: 
    product_data = data[data['product_identifier'] == product_id]

    # Find outliers for the 'sales' column of the filtered data
    outliers = find_outliers_IQR(product_data['sales'])

    # Print the results
    print('Product ID:', product_id)
    print('Number of outliers:', len(outliers))
    print('Max outlier value:', outliers.max())
    print('Min outlier value:', outliers.min())
    print('% of outliers:', len(outliers)/len(product_data)*100,'\n')


    product_data.boxplot(by ='product_identifier', column =['sales'],figsize=(15,15), grid = False)
    plt.title(f'Boxplot of sale of product {product_id}')
    plt.show()


# In[122]:


def find_outliers_IQR(x):
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    IQR = q3 - q1
    outliers = x[((x< (q1 - 1.5 * IQR)) | (x > (q3 + 1.5 * IQR)))]

    return outliers

# Set the product identifier for which you want to find outliers
product_ids = [  74,  337,  423,  432,  581,  611,  631,  659,  743,  797,  868,904,  926,  972,  973, 1054, 1135, 1173, 1190, 1196, 1228, 1240,1242, 1275, 1322,
               1328, 1365, 1424, 1472, 1508, 1542, 1548, 1599,1629, 1672, 1694, 1727, 1753, 2294, 2332, 2492, 2768, 2794, 2818,2853, 2932, 2935, 3004, 3008, 3021]

for product_id in product_ids: 
    product_data = data[data['product_identifier'] == product_id]

    # Find outliers for the 'sales' column of the filtered data
    outliers = find_outliers_IQR(product_data['sales'])

    # Print the results
    print('Product ID:', product_id)
    print('Number of outliers:', len(outliers))
    print('Max outlier value:', outliers.max())
    print('Min outlier value:', outliers.min())
    print('% of outliers:', len(outliers)/len(product_data)*100,'\n')


    product_data.boxplot(by ='product_identifier', column =['sell_price'],figsize=(15,15), grid = False)
    plt.title(f'Boxplot of sell price of product {product_id}')
    plt.show()


# # Skewness Detection

# In[123]:


data[continous_features].agg(['skew', 'kurtosis']).transpose()


# In[124]:


y_drinks_and_food.skew()


# In[125]:


y_fast_moving_consumer_goods.skew()


# In[126]:


y_others.skew()


# In[127]:


for i in sales_monthly:
    print(f'skewness of {i} product ( {sales_monthly[i].skew()} )')
   
    if sales_monthly[i].skew() > 1.5 or sales_monthly[i].skew() < -1.5 :
        print( f'product {i} is highly skewed \n  ')


# # Box-Cox Transformation

# In[128]:


cols = [1542]
#sales_monthly1=sales_monthly.copy()
for i in cols:
    sales_monthly[i] = sales_monthly[i].apply(lambda x: np.power(x, (1/1.5)))
    print(sales_monthly[i].skew())


# In[129]:


cols = [ 2294]
for i in cols:
    sales_monthly[i] = sales_monthly[i].apply(lambda x: np.power(x, (1/5)))
    print(sales_monthly[i].skew())


# In[130]:


cols = [ 926]
for i in cols:
    sales_monthly[i] = sales_monthly[i].apply(lambda x: np.power(x, (1/0.6)))
    print(sales_monthly[i].skew())


# In[131]:


cols = [1240,1672]
for i in cols:
    sales_monthly[i] = sales_monthly[i].apply(lambda x: np.power(x, (1/0.8)))
    print(sales_monthly[i].skew())


# In[132]:


for i in sales_monthly:
    print(f'skewness of {i} product ( {sales_monthly[i].skew()} )')
   
    if sales_monthly[i].skew() > 1.5 or sales_monthly[i].skew() < -1.5 :
        print( f'product {i} is highly skewed \n  ')


# In[133]:


sales_monthly.describe().transpose()


# # Distribution of numerical data

# In[134]:


for i in continous_features:
    sns.distplot(data[i])
    plt.xlabel(i)
    plt.title(i)
    plt.show()


# In[135]:


for i in sales_monthly.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=sales_monthly, x=i)
    plt.xlabel(i)
    plt.title("Distribution of " +str(i))
    plt.show()


# # Encoding category wise data

# In[136]:


drinks_and_food_en = pd.get_dummies(drinks_and_food, columns=['category_of_product', 'state'])


# In[137]:


fast_moving_consumer_goods_en = pd.get_dummies(fast_moving_consumer_goods, columns=['category_of_product', 'state'])


# In[138]:


others_en = pd.get_dummies(others, columns=['category_of_product', 'state'])


# # -----------------------------------------------------------------------------------------

# # Comparing Other Categories

# In[139]:


dfood1 = pd.DataFrame({'Order Date': y_drinks_and_food.index , 'Sales':y_drinks_and_food.values})


# In[140]:


fmcg1 = pd.DataFrame({'Order Date': y_fast_moving_consumer_goods.index , 'Sales':y_fast_moving_consumer_goods.values})


# In[141]:


others1 = pd.DataFrame({'Order Date': y_others.index , 'Sales':y_others.values})


# # Data Exploration

# In[142]:


store_drinks_fmcg = dfood1.merge(fmcg1, how='inner', on='Order Date')
store_drinks_fmcg.rename(columns={'Sales_x':'drinks_&_food_sales','Sales_y':'fmcg_sales'},inplace=True)


# In[143]:


store_drinks_others = dfood1.merge(others1, how='inner', on='Order Date')
store_drinks_others.rename(columns={'Sales_x':'drinks_&_food_sales','Sales_y':'others_sales'},inplace=True)


# In[144]:


store_others_fmcg = others1.merge(fmcg1, how='inner', on='Order Date')
store_others_fmcg.rename(columns={'Sales_x':'others','Sales_y':'fmcg_sales'},inplace=True)


# In[145]:


store_drinks_fmcg_others = others1.merge(store_drinks_fmcg, how='inner', on='Order Date')
store_drinks_fmcg_others.rename(columns={'Sales':'others_sales','Sales_x':'Drinks_&_food_sales','Sales_y':'FMCG sales'},inplace=True)


# ## Plotting

# In[146]:


plt.figure(figsize=(20,8))
plt.plot(store_drinks_fmcg['Order Date'],store_drinks_fmcg['drinks_&_food_sales'],'b-',label='drinks_&_food_sales',linewidth=5)
plt.plot(store_drinks_fmcg['Order Date'],store_drinks_fmcg['fmcg_sales'],'r-',label='fmcg_sales',linewidth=5)
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales of drinks and FMCG')
plt.legend()


# In[147]:


plt.figure(figsize=(20,8))
plt.plot(store_drinks_others['Order Date'],store_drinks_others['drinks_&_food_sales'],'b-',label='drinks_&_food_sales',linewidth=5)
plt.plot(store_drinks_others['Order Date'],store_drinks_others['others_sales'],'r-',label='others sales',linewidth=5)
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales of drinks and other supplies')
plt.legend()


# In[148]:


plt.figure(figsize=(20,8))
plt.plot(store_others_fmcg['Order Date'],store_others_fmcg['fmcg_sales'],'b-',label='fmcg_sales',linewidth=5)
plt.plot(store_others_fmcg['Order Date'],store_others_fmcg['others'],'r-',label='others',linewidth=5)
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales of FMCG and Other supplies')
plt.legend()


# In[149]:


plt.figure(figsize=(20,8))
plt.plot(store_drinks_fmcg_others['Order Date'],store_drinks_fmcg_others['fmcg_sales'],'b-',label='furniture',linewidth=5)
plt.plot(store_drinks_fmcg_others['Order Date'],store_drinks_fmcg_others['drinks_&_food_sales'],'r-',label='drinks_&_food_sales',linewidth=5)
plt.plot(store_drinks_fmcg_others['Order Date'],store_drinks_fmcg_others['others_sales'],'g-',label='others_sales',linewidth=5)
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales of Drinks & Food ,FMCG and Other supllies')
plt.legend()


# # -----------------------------------------------------------------------------

# # -----------------------------------------------------------------------------

# # MODEL CREATION

# # SARIMA Model

# NOTE: During earlier ETS decomposition, it was observed that both category-wise and product-wise data exhibit seasonality. Therefore, SARIMA was chosen over ARIMA to account for the seasonal component in the data.

# ## Product Wise Modelling

# ### Choosing Best Model

# In[150]:


from statsmodels.tsa.stattools import adfuller
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

def adf_test(series, title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    result = adfuller  (series.dropna(), autolag='AIC')  # .dropna() handles differenced data

    labels = ['ADF test statistic', 'p-value', '# lags used', '# observations']
    out = pd.Series(result[0:4], index=labels)

    for key, val in result[4].items():
        out[f'critical value ({key})'] = val

    print(out.to_string())  # .to_string() removes the line "dtype: float64"

    if result[1] <= 0.05:
        d = 0
        print('Data is stationary')
    else:
        d = 1
        print('Data is not stationary')

    return d


# Assuming sales_monthly is a DataFrame with columns representing time series data

for col in sales_monthly.columns:
    d = adf_test(sales_monthly[col])
    
    exog_var = price_monthly[col]

    p = q = range(0, 2)
    pdq = list(itertools.product(p, [d], q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, [d], q))]

    lowest_aic = float('inf')
    best_pdq = None
    best_seasonal_pdq = None

    print(f'\nARIMA for: {col}\n')
    y = sales_monthly.loc[:, [col]]

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y.values.flatten(), order=param, seasonal_order=param_seasonal, enforce_invertibility=False)
                results = mod.fit()

                if results.aic < lowest_aic:
                    lowest_aic = results.aic
                    best_pdq = param
                    best_seasonal_pdq = param_seasonal

                print(f'ARIMA{param}x{param_seasonal} - AIC: {results.aic:.2f}')

            except:
                continue

    print(f'Best ARIMA{best_pdq}x{best_seasonal_pdq} - AIC: {lowest_aic:.2f}')

    mod = sm.tsa.statespace.SARIMAX(y.values.flatten(), order=best_pdq, seasonal_order=best_seasonal_pdq, enforce_invertibility=False)
    results = mod.fit()
    print(results.summary().tables[1])

    results.plot_diagnostics(figsize=(15, 8))
    plt.show()


# ### Prediction using selected model

# In[151]:


from statsmodels.tsa.stattools import adfuller
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

def adf_test(series, title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    result = adfuller  (series.dropna(), autolag='AIC')  # .dropna() handles differenced data

    #labels = ['ADF test statistic', 'p-value', '# lags used', '# observations']
    #out = pd.Series(result[0:4], index=labels)

    #for key, val in result[4].items():
        #out[f'critical value ({key})'] = val

    #print(out.to_string())  # .to_string() removes the line "dtype: float64"

    if result[1] <= 0.05:
        d = 0
        print('Data is stationary')
    else:
        d = 1
        print('Data is not stationary')

    return d


# Assuming sales_monthly is a DataFrame with columns representing time series data

for col in sales_monthly.columns:
    print(f'\nARIMA for: {col}\n')
    d = adf_test(sales_monthly[col])
    exog_var = price_monthly[col]

    p = q = range(0, 2)
    pdq = list(itertools.product(p, [d], q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, [d], q))]

    lowest_aic = float('inf')
    best_pdq = None
    best_seasonal_pdq = None

    y = sales_monthly.loc[:, [col]]

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y.values.flatten(), order=param, seasonal_order=param_seasonal, enforce_invertibility=False)
                results = mod.fit()

                if results.aic < lowest_aic:
                    lowest_aic = results.aic
                    best_pdq = param
                    best_seasonal_pdq = param_seasonal

                #print(f'ARIMA{param}x{param_seasonal} - AIC: {results.aic:.2f}')

            except:
                continue

    print(f'Best ARIMA{best_pdq}x{best_seasonal_pdq} - AIC: {lowest_aic:.2f}\n')

    mod = sm.tsa.statespace.SARIMAX(y, order=best_pdq, seasonal_order=best_seasonal_pdq, enforce_invertibility=False)
    results = mod.fit()
    #print(results.summary().tables[1])

    #results.plot_diagnostics(figsize=(15, 8))
    #plt.show()
    # Generate prediction plot
    pred = results.get_prediction(start='2013-07-01', dynamic=False)
    pred_ci = pred.conf_int()

    ax = y['2012':].plot(label='Observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Date')
    ax.set_ylabel(col)

    plt.legend()
    plt.show()


# ### Evaluation of Model Created

# In[152]:


from statsmodels.tsa.stattools import adfuller
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

def adf_test(series, title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    result = adfuller  (series.dropna(), autolag='AIC')  # .dropna() handles differenced data

    #labels = ['ADF test statistic', 'p-value', '# lags used', '# observations']
    #out = pd.Series(result[0:4], index=labels)

    #for key, val in result[4].items():
        #out[f'critical value ({key})'] = val

    #print(out.to_string())  # .to_string() removes the line "dtype: float64"

    if result[1] <= 0.05:
        d = 0
        #print('Data is stationary')
    else:
        d = 1
        #print('Data is not stationary')

    return d


# Assuming sales_monthly is a DataFrame with columns representing time series data
SA_data = pd.DataFrame(columns=['Product Identifier','MSE_sarima', 'RMSE_sarima','MAE_sarima'])

for col in sales_monthly.columns:
    #print(f'\nARIMA for: {col}\n')
    d = adf_test(sales_monthly[col])
    exog_var = price_monthly[col]

    p = q = range(0, 2)
    pdq = list(itertools.product(p, [d], q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, [d], q))]

    lowest_aic = float('inf')
    best_pdq = None
    best_seasonal_pdq = None

    y = sales_monthly.loc[:, [col]]

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y.values.flatten(), order=param, seasonal_order=param_seasonal, enforce_invertibility=False)
                results = mod.fit()

                if results.aic < lowest_aic:
                    lowest_aic = results.aic
                    best_pdq = param
                    best_seasonal_pdq = param_seasonal

                #print(f'ARIMA{param}x{param_seasonal} - AIC: {results.aic:.2f}')

            except:
                continue

    #print(f'Best ARIMA{best_pdq}x{best_seasonal_pdq} - AIC: {lowest_aic:.2f}\n')

    mod = sm.tsa.statespace.SARIMAX(y, order=best_pdq, seasonal_order=best_seasonal_pdq, enforce_invertibility=False)
    results = mod.fit()
    #print(results.summary().tables[1])

    #results.plot_diagnostics(figsize=(15, 8))
    #plt.show()
    # Generate prediction plot
    pred = results.get_prediction(start='2013-07-01', dynamic=False)
    pred_ci = pred.conf_int()

   # ax = y['2012':].plot(label='Observed')
   # pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
   # ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)

    #ax.set_xlabel('Date')
    #ax.set_ylabel(col)

    #plt.legend()
    #plt.show()
    
    #y_forecasted = pred.predicted_mean
    #y_truth = y['2013-07-01':]
    #y_truth= y_truth.stack()
    #mse = ((y_forecasted - y_truth) ** 2).mean()
    #print("The Mean Squared Error of our forecasts is {}".format(round(mse, 2)))
    #print("The Root Mean Squared Error of our forecasts is {}".format(round(np.sqrt(mse))))
     # Calculate MSE and RMSE
    y_forecasted = pred.predicted_mean
    y_truth = y['2013-07-01':]
    y_pred = y_truth.stack()
    mse = ((y_forecasted - y_pred) ** 2).mean()
    rmse = np.sqrt(mse)
    mae=mean_absolute_error(y_forecasted,y_pred)

    # Store MSE and RMSE values in the dataset
    SA_data = SA_data.append({'Product Identifier': col, 'MSE_sarima': mse, 'RMSE_sarima': rmse, 'MAE_sarima': mae}, ignore_index=True)


# In[153]:


SA_data


# ### Forecasting 

# In[154]:


from statsmodels.tsa.stattools import adfuller
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

def adf_test(series, title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    result = adfuller  (series.dropna(), autolag='AIC')  # .dropna() handles differenced data

    #labels = ['ADF test statistic', 'p-value', '# lags used', '# observations']
    #out = pd.Series(result[0:4], index=labels)

    #for key, val in result[4].items():
        #out[f'critical value ({key})'] = val

    #print(out.to_string())  # .to_string() removes the line "dtype: float64"

    if result[1] <= 0.05:
        d = 0
        print('Data is stationary')
    else:
        d = 1
        print('Data is not stationary')

    return d


# Assuming sales_monthly is a DataFrame with columns representing time series data

for col in sales_monthly.columns:
    print(f'\nARIMA for: {col}\n')
    d = adf_test(sales_monthly[col])
    exog_var = price_monthly[col]

    p = q = range(0, 2)
    pdq = list(itertools.product(p, [d], q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, [d], q))]

    lowest_aic = float('inf')
    best_pdq = None
    best_seasonal_pdq = None

    y = sales_monthly.loc[:, [col]]

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y.values.flatten(), order=param, seasonal_order=param_seasonal, enforce_invertibility=False)
                results = mod.fit()

                if results.aic < lowest_aic:
                    lowest_aic = results.aic
                    best_pdq = param
                    best_seasonal_pdq = param_seasonal

                #print(f'ARIMA{param}x{param_seasonal} - AIC: {results.aic:.2f}')

            except:
                continue

    print(f'Best ARIMA{best_pdq}x{best_seasonal_pdq} - AIC: {lowest_aic:.2f}\n')

    mod = sm.tsa.statespace.SARIMAX(y, order=best_pdq, seasonal_order=best_seasonal_pdq, enforce_invertibility=False)
    results = mod.fit()
    #print(results.summary().tables[1])

    #results.plot_diagnostics(figsize=(15, 8))
    #plt.show()
    # Generate prediction plot
    pred = results.get_prediction(start='2013-07-01', dynamic=False)
    pred_ci = pred.conf_int()

    ax = y['2012':].plot(label='Observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Date')
    ax.set_ylabel(col)

    plt.legend()
    plt.show()
    
    #y_forecasted = pred.predicted_mean
    #y_truth = y['2013-07-01':]
    #y_truth= y_truth.stack()
    #mse = ((y_forecasted - y_truth) ** 2).mean()
    #print("The Mean Squared Error of our forecasts is {}".format(round(mse, 2)))
    #print("The Root Mean Squared Error of our forecasts is {}".format(round(np.sqrt(mse))))
    
    
    pred_uc = results.get_forecast(steps=100)
    pred_cl = pred_uc.conf_int()
    ax = y.plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_cl.index, pred_cl.iloc[:, 0], pred_cl.iloc[:, 1], color='k', alpha=.2)
    ax.set_ylim([-100, 100])
    ax.set_xlabel('Date')
    ax.set_ylabel(col)
    plt.legend()
    plt.show()


# ## Category wise Modelling

# ### Drinks and Food

# In[155]:


y_drinks_and_food.skew()


# In[156]:


#checking stationarity


# In[157]:


from statsmodels.tsa.stattools import adfuller

def adf_test1(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


# In[158]:


adf_test1(y_drinks_and_food)


# In[159]:


import itertools
import statsmodels.api as sm

p = q = range(0, 2)
d = [0]  # Fixing d to zero
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

lowest_aic = float('inf')
best_pdq = None
best_seasonal_pdq = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y_drinks_and_food,
                                           order=param,
                                           seasonal_order=param_seasonal,
                                           #enforce_stationarity=False,
                                           enforce_invertibility=False)

            results = mod.fit()
                
            if results.aic < lowest_aic:
                lowest_aic = results.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal

            print('ARIMA{}x{}12 - AIC: {}'.format(param, param_seasonal, results.aic))

        except:
            continue

print(f'Best ARIMA{best_pdq}x{best_seasonal_pdq} - AIC: {lowest_aic:.2f}')


# In[160]:


mod = sm.tsa.statespace.SARIMAX(y_drinks_and_food,
                                    order=best_pdq,
                                    seasonal_order=best_seasonal_pdq,
                                    #enforce_stationarity=False,
                                    enforce_invertibility=False)
results = mod.fit()


results = mod.fit()
print (results.summary().tables[1]) 
results.plot_diagnostics (figsize=(16, 8)) 
plt.show()


# In[161]:


pred=results.get_prediction (start='2013-07-01', dynamic=False)

pred_ci = pred.conf_int()

ax=y_drinks_and_food[ '2012':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                 pred_ci.iloc[:, 0],
                 pred_ci.iloc[:, 1], color='k', alpha=.2)


ax.set_xlabel('Date')

ax.set_ylabel('drinks and food Sales')

plt.legend()

plt.show()


# In[162]:


y_forecasted_df=pred.predicted_mean

y_truth_df = y_drinks_and_food[ '2013-07-01':]

SA_errors_df = pd.DataFrame(index=y_truth_df.index)
SA_errors_df['Modelname'] = 'SARIMA'
SA_errors_df['Actual'] = y_truth_df
SA_errors_df['Predicted'] = y_forecasted_df
SA_errors_df['Error'] = y_forecasted_df - y_truth_df
SA_errors_df.head()

mse_sa_df = ((y_forecasted_df - y_truth_df) ** 2).mean()

# Calculate the mean absolute error (MAE)
mae_sa_df = mean_absolute_error(y_truth_df, y_forecasted_df)

# Calculate the mean squared error (MSE)
mse_sa_df = mean_squared_error(y_truth_df, y_forecasted_df)
# Calculate the root mean squared error (RMSE)
rmse_sa_df = np.sqrt(mse_sa_df)

print("The Mean Squared Error of our forecasts is {}".format(mse_sa_df))
print("The Root Mean Squared Error of our forecasts is {}".format(rmse_sa_df))


# In[163]:


pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = y_drinks_and_food.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_ylim([300, 600])
ax.set_xlabel('Date')
ax.set_ylabel(' Sales')
plt.legend()
plt.show()


# ## Fast Moving Consumer Goods

# In[164]:


y_fast_moving_consumer_goods.skew()


# In[165]:


#checking stationarity


# In[166]:


adf_test1(y_fast_moving_consumer_goods)


# In[167]:


import itertools
import statsmodels.api as sm

p = q = range(0, 2)
d = [1]  # Fixing d to One
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

lowest_aic = float('inf')
best_pdq = None
best_seasonal_pdq = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y_fast_moving_consumer_goods,
                                           order=param,
                                           seasonal_order=param_seasonal,
                                           #enforce_stationarity=False,
                                           enforce_invertibility=False)

            results = mod.fit()
                
            if results.aic < lowest_aic:
                lowest_aic = results.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal

            print('ARIMA{}x{}12 - AIC: {}'.format(param, param_seasonal, results.aic))

        except:
            continue

print(f'Best ARIMA{best_pdq}x{best_seasonal_pdq} - AIC: {lowest_aic:.2f}')


# In[168]:


mod = sm.tsa.statespace.SARIMAX(y_fast_moving_consumer_goods,
                                    order=best_pdq,
                                    seasonal_order=best_seasonal_pdq,
                                    #enforce_stationarity=False,
                                    enforce_invertibility=False)
results = mod.fit()


results = mod.fit()
print (results.summary().tables[1]) 
results.plot_diagnostics (figsize=(16, 8)) 
plt.show()


# In[169]:


pred=results.get_prediction (start='2013-07-01', dynamic=False)

pred_ci = pred.conf_int()

ax=y_fast_moving_consumer_goods[ '2012':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                 pred_ci.iloc[:, 0],
                 pred_ci.iloc[:, 1], color='k', alpha=.2)


ax.set_xlabel('Date')

ax.set_ylabel('fast moving consumer goods Sales')

plt.legend()

plt.show()


# In[170]:


y_forecasted_fmcg=pred.predicted_mean

y_truth_fmcg = y_fast_moving_consumer_goods[ '2013-07-01':]

SA_errors_fmcg = pd.DataFrame(index=y_truth_fmcg.index)
SA_errors_fmcg['Modelname'] = 'SARIMA'
SA_errors_fmcg['Actual'] = y_truth_fmcg
SA_errors_fmcg['Predicted'] = y_forecasted_fmcg
SA_errors_fmcg['Error'] = y_forecasted_fmcg - y_truth_fmcg
SA_errors_fmcg.head()

mse_sa_fmcg = ((y_forecasted_fmcg - y_truth_fmcg) ** 2).mean()

# Calculate the mean absolute error (MAE)
mae_sa_fmcg = mean_absolute_error(y_truth_fmcg, y_forecasted_fmcg)

# Calculate the mean squared error (MSE)
mse_sa_fmcg = mean_squared_error(y_truth_fmcg, y_forecasted_fmcg)
# Calculate the root mean squared error (RMSE)
rmse_sa_fmcg = np.sqrt(mse_sa_fmcg)

print("The Mean Squared Error of our forecasts is {}".format(mse_sa_fmcg))
print("The Root Mean Squared Error of our forecasts is {}".format(rmse_sa_fmcg))


# In[171]:


pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = y_fast_moving_consumer_goods.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_ylim([-100, 300])
ax.set_xlabel('Date')
ax.set_ylabel(' Sales')
plt.legend()
plt.show()


# ## Others

# In[172]:


y_others.skew()


# In[173]:


#Checking stationarity
adf_test1(y_others)


# In[174]:


import itertools
import statsmodels.api as sm

p = q = range(0, 2)
d = [1]  # Fixing d to One
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

lowest_aic = float('inf')
best_pdq = None
best_seasonal_pdq = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y_others,
                                           order=param,
                                           seasonal_order=param_seasonal,
                                           #enforce_stationarity=False,
                                           enforce_invertibility=False)

            results = mod.fit()
                
            if results.aic < lowest_aic:
                lowest_aic = results.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal

            print('ARIMA{}x{}12 - AIC: {}'.format(param, param_seasonal, results.aic))

        except:
            continue

print(f'Best ARIMA{best_pdq}x{best_seasonal_pdq} - AIC: {lowest_aic:.2f}')


# In[175]:


mod = sm.tsa.statespace.SARIMAX(y_others,
                                    order=best_pdq,
                                    seasonal_order=best_seasonal_pdq,
                                    #enforce_stationarity=False,
                                    enforce_invertibility=False)
results = mod.fit()


results = mod.fit()
print (results.summary().tables[1]) 
results.plot_diagnostics (figsize=(16, 8)) 
plt.show()


# In[176]:


pred=results.get_prediction (start='2013-07-01', dynamic=False)

pred_ci = pred.conf_int()

ax=y_others[ '2012':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                 pred_ci.iloc[:, 0],
                 pred_ci.iloc[:, 1], color='k', alpha=.2)


ax.set_xlabel('Date')

ax.set_ylabel('others Sales')

plt.legend()

plt.show()


# In[177]:


y_forecasted_others=pred.predicted_mean

y_truth_others = y_others[ '2013-07-01':]

SA_errors_others = pd.DataFrame(index=y_truth_others.index)
SA_errors_others['Modelname'] = 'SARIMA'
SA_errors_others['Actual'] = y_truth_others
SA_errors_others['Predicted'] = y_forecasted_others
SA_errors_others['Error'] = y_forecasted_others - y_truth_others
SA_errors_others.head()

mse_sa_others = ((y_forecasted_others - y_truth_others) ** 2).mean()

# Calculate the mean absolute error (MAE)
mae_sa_others = mean_absolute_error(y_truth_others, y_forecasted_others)

# Calculate the mean squared error (MSE)
mse_sa_others = mean_squared_error(y_truth_others, y_forecasted_others)
# Calculate the root mean squared error (RMSE)
rmse_sa_others = np.sqrt(mse_sa_others)

print("The Mean Squared Error of our forecasts is {}".format(mse_sa_others))
print("The Root Mean Squared Error of our forecasts is {}".format(rmse_sa_others))


# In[178]:


pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = y_others.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_ylim([-50, 50])
ax.set_xlabel('Date')
ax.set_ylabel(' Sales')
plt.legend()
plt.show()


# # ---------------------------------------------------------------------------------------

# # Time series modelling with prophet

# In[179]:


get_ipython().system('pip install prophet')


# In[180]:


import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly
from prophet import Prophet


# ## Product wise-Prophet

# In[181]:


train_pro = sales_monthly.iloc[:]
#add 20 if need to do evaluation


# In[182]:


for col in sales_monthly:
    print(f'\nPROPHET method for {col}\n')

    y_df = train_pro.loc[:, [col]]
    y_df['ds'] = y_df.index
    y_df.columns = ['y', 'ds']

    m_y = Prophet()
    m_y.fit(y_df)

    future_y = m_y.make_future_dataframe(periods=12, freq='MS')

    forecast_y = m_y.predict(future_y)

    forecast_y[['ds', 'yhat_lower', 'yhat_upper', 'yhat']]

    plt.figure()
    m_y.plot(forecast_y)
    plt.title(f'Sales of {col}')
    plt.xlabel('Years')
    plt.ylabel('Sales')

    plt.figure()
    plot_plotly(m_y, forecast_y)

    plt.figure(figsize=(12, 5))
    forecast_y.plot(x='ds', y='yhat')
    plt.show()

    plt.figure()
    m_y.plot_components(forecast_y)

    plt.figure()
    plot_components_plotly(m_y, forecast_y)


# In[183]:


from sklearn.metrics import mean_squared_error
import numpy as np
prophet_data = pd.DataFrame(columns=['Product Identifier', 'MSE_p', 'RMSE_p','MAE_p'])
for col in sales_monthly:
    print(f'\nPROPHET method for {col}\n')

    y_df = train_pro.loc[:, [col]]
    y_df['ds'] = y_df.index
    y_df.columns = ['y', 'ds']

    m_y = Prophet()
    m_y.fit(y_df)

    future_y = m_y.make_future_dataframe(periods=12, freq='MS')

    forecast_y = m_y.predict(future_y)

    forecast_y[['ds', 'yhat_lower', 'yhat_upper', 'yhat']]

    # Calculate MSE and RMSE
    actual_values = y_df['y'].values
    predicted_values = forecast_y.loc[:len(actual_values)-1, 'yhat'].values
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    mae= mean_absolute_error(actual_values,predicted_values)
    
    prophet_data = prophet_data.append({'Product Identifier': col, 'MSE_p': mse, 'RMSE_p': rmse,'MAE_p': mae}, ignore_index=True)

    # Rest of the code...
    
    prophet_data.drop_duplicates(subset='Product Identifier', inplace=True)
    #prophet_data=prophet_data[1:]


# In[184]:


#prophet_data.drop(0, inplace=True)
prophet_data


# # -----------------------------------------------------------------------------------------------

# # Category Wise Modelling

# ## Drinks and Food

# In[185]:


y_drinks_and_food_df = y_drinks_and_food.to_frame()
y_drinks_and_food_df['ds'] = y_drinks_and_food_df.index
y_drinks_and_food_df.columns = ['y','ds']
y_drinks_and_food_df


# In[186]:


y_drinks_and_food_df.info()


# In[187]:


m_drinks_and_food = Prophet()
m_drinks_and_food.fit(y_drinks_and_food_df)


# In[188]:


future_drinks_and_food = m_drinks_and_food.make_future_dataframe(periods=24,freq='MS')
future_drinks_and_food


# In[189]:


forecast_drinks_and_food = m_drinks_and_food.predict(future_drinks_and_food)
forecast_drinks_and_food


# In[190]:


forecast_drinks_and_food.columns


# In[191]:


forecast_drinks_and_food[['ds','yhat_lower', 'yhat_upper','yhat']].tail(12)


# In[194]:


y_drinks_and_food


# In[195]:


actual_values_df = y_drinks_and_food_df['y'].values
predicted_values_df = forecast_drinks_and_food.loc[:len(actual_values_df)-1, 'yhat'].values

pr_errors_df = pd.DataFrame(index=np.arange(len(actual_values_df)))
pr_errors_df['Modelname'] = 'Prophet'
pr_errors_df['Actual'] = actual_values_df
pr_errors_df['Predicted'] = predicted_values_df
pr_errors_df['Error'] = predicted_values_df - actual_values_df

# Calculate the mean squared error (MSE)
mse_pr_df = mean_squared_error(actual_values_df, predicted_values_df)

# Calculate the mean absolute error (MAE)
mae_pr_df = mean_absolute_error(actual_values_df, predicted_values_df)

# Calculate the root mean squared error (RMSE)
rmse_pr_df = np.sqrt(mse_pr_df)


# In[196]:


m_drinks_and_food.plot(forecast_drinks_and_food);
plt.title('Sales of drinks_and_food')
plt.xlabel('Years')
plt.ylabel('Sales')


# In[197]:


from prophet.plot import plot_plotly, plot_components_plotly
plot_plotly(m_drinks_and_food, forecast_drinks_and_food)
forecast_drinks_and_food.plot(x='ds',y='yhat',figsize=(12,5))


# In[198]:


m_drinks_and_food.plot_components(forecast_drinks_and_food);


# In[199]:


plot_components_plotly(m_drinks_and_food , forecast_drinks_and_food)


# ## Fast Moving Consumer Goods

# In[200]:


y_fast_moving_consumer_goods_df = y_fast_moving_consumer_goods.to_frame()
y_fast_moving_consumer_goods_df['ds'] = y_fast_moving_consumer_goods_df.index
y_fast_moving_consumer_goods_df.columns = ['y','ds']
y_fast_moving_consumer_goods_df


# In[201]:


y_fast_moving_consumer_goods_df.info()


# In[202]:


m_fast_moving_consumer_goods = Prophet()
m_fast_moving_consumer_goods.fit(y_fast_moving_consumer_goods_df)


# In[203]:


future_fast_moving_consumer_goods = m_fast_moving_consumer_goods.make_future_dataframe(periods=24,freq='MS')
future_fast_moving_consumer_goods


# In[204]:


forecast_fast_moving_consumer_goods = m_fast_moving_consumer_goods.predict(future_fast_moving_consumer_goods)
forecast_fast_moving_consumer_goods


# In[205]:


forecast_fast_moving_consumer_goods.columns


# In[206]:


forecast_fast_moving_consumer_goods[['ds','yhat_lower', 'yhat_upper','yhat']].tail(12)


# In[207]:


actual_values_fmcg = y_fast_moving_consumer_goods_df['y'].values
predicted_values_fmcg = forecast_fast_moving_consumer_goods.loc[:len(actual_values_fmcg)-1, 'yhat'].values

pr_errors_fmcg = pd.DataFrame(index=np.arange(len(actual_values_fmcg)))
pr_errors_fmcg['Modelname'] = 'Prophet'
pr_errors_fmcg['Actual'] = actual_values_fmcg
pr_errors_fmcg['Predicted'] = predicted_values_fmcg
pr_errors_fmcg['Error'] = predicted_values_fmcg - actual_values_fmcg

# Calculate the mean squared error (MSE)
mse_pr_fmcg = mean_squared_error(actual_values_fmcg, predicted_values_fmcg)

# Calculate the mean absolute error (MAE)
mae_pr_fmcg = mean_absolute_error(actual_values_fmcg, predicted_values_fmcg)

# Calculate the root mean squared error (RMSE)
rmse_pr_fmcg = np.sqrt(mse_pr_fmcg)


# In[208]:


m_fast_moving_consumer_goods.plot(forecast_fast_moving_consumer_goods);
plt.title('Sales of fast_moving_consumer_goods')
plt.xlabel('Years')
plt.ylabel('Sales')


# In[209]:


from prophet.plot import plot_plotly, plot_components_plotly
plot_plotly(m_fast_moving_consumer_goods, forecast_fast_moving_consumer_goods)
forecast_fast_moving_consumer_goods.plot(x='ds',y='yhat',figsize=(12,5))


# In[210]:


m_fast_moving_consumer_goods.plot_components(forecast_fast_moving_consumer_goods);


# In[211]:


plot_components_plotly(m_fast_moving_consumer_goods , forecast_fast_moving_consumer_goods)


# ## Others

# In[212]:


y_others_df = y_others.to_frame()
y_others_df['ds'] = y_others_df.index
y_others_df.columns = ['y','ds']
y_others_df


# In[213]:


y_others_df.info()


# In[214]:


m_others = Prophet()
m_others.fit(y_others_df)


# In[215]:


future_others = m_others.make_future_dataframe(periods=24,freq='MS')
future_others


# In[216]:


forecast_others = m_others.predict(future_others)
forecast_others


# In[217]:


forecast_others.columns


# In[218]:


forecast_others[['ds','yhat_lower', 'yhat_upper','yhat']].tail(12)


# In[219]:


actual_values_others = y_others_df['y'].values
predicted_values_others= forecast_others.loc[:len(actual_values_others)-1, 'yhat'].values

pr_errors_others = pd.DataFrame(index=np.arange(len(actual_values_others)))
pr_errors_others['Modelname'] = 'Prophet'
pr_errors_others['Actual'] = actual_values_others
pr_errors_others['Predicted'] = predicted_values_others
pr_errors_others['Error'] = predicted_values_others - actual_values_others

# Calculate the mean squared error (MSE)
mse_pr_others = mean_squared_error(actual_values_others, predicted_values_others)

# Calculate the mean absolute error (MAE)
mae_pr_others= mean_absolute_error(actual_values_others, predicted_values_others)

# Calculate the root mean squared error (RMSE)
rmse_pr_others = np.sqrt(mse_pr_others)


# In[220]:


m_others.plot(forecast_others);
plt.title('Sales of others')
plt.xlabel('Years')
plt.ylabel('Sales')


# In[221]:


from prophet.plot import plot_plotly, plot_components_plotly
plot_plotly(m_others, forecast_others)
forecast_others.plot(x='ds',y='yhat',figsize=(12,5))


# In[222]:


m_others.plot_components(forecast_others);


# In[223]:


plot_components_plotly(m_others , forecast_others)


# # ---------------------------------------------------------------------------------------

# NOTE: simple exponential smoothing is not suitable for seasonal data.So we didn't use it and Triple exponential smoothing need Minimum 24 data entries we only have 25 so we drop it.

# # Double-Exponential Smoothing

# ## Product Wise Modelling

# ###### Spliting data (Train=80%, Test= 20%)

# In[224]:


# Split the time series data (Train-20, Test-6)
#retail_data = retail_data.set_index('StartDate')
print('Total records in dataset:', len(sales_monthly))
sales_train = sales_monthly.iloc[0:20]               
sales_test = sales_monthly.iloc[20:]

sales_pred_train = sales_monthly.iloc[0:20]               
sales_pred_test = sales_monthly.iloc[20:]
print('Total records in Training set:', len(sales_train))
print('Total records in Test set:', len(sales_test))


# ###### Plot Train and Test data

# In[225]:


import matplotlib.pyplot as plt

for col in sales_monthly:
    sales_train[col].plot(legend=True, label='TRAIN (80%)')
    sales_test[col].plot(legend=True, label='TEST (20%)', figsize=(12, 8))
    plt.title(f'Sales of {col}')
    plt.xlabel('Index')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()


# In[226]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

for col in sales_train:
    double_model = ExponentialSmoothing(sales_train[col], trend='add').fit()
    doublemodel_preds = double_model.forecast(6).rename('DES Forecast')
    
    # Print the forecasted values
    print(f'\nDouble Exponential Smoothing (DES) forecast for {col}:\n')
    #print(doublemodel_preds)

    des_errors_df = sales_test[[col]].copy()
    des_errors_df['Predicted_sales'] = doublemodel_preds
    des_errors_df['Error'] = doublemodel_preds - sales_test[col]
    des_errors_df.insert(0, 'Modelname', 'Holtman-DES')
    #print(f'\nDES Errors for {col}:\n')
    #print(des_errors_df.head())


# Evaluate predictions for Holt Winters-Double Exponential Smoothing
    fig = plt.figure(figsize=(14,7))
    plt.plot(sales_train.index, sales_train[col], label='Train')
    plt.plot(sales_test.index, sales_test[col], label='Test')
    plt.plot(des_errors_df.index, des_errors_df['Predicted_sales'], label='Forecast - HW-DES')
    plt.legend(loc='best')
    plt.xlabel('StartDate')
    plt.ylabel('sales')
    plt.title('Forecast using Holt Winters-Double Exponential Smoothing')
    plt.show()


# In[227]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing
des_data = pd.DataFrame(columns=['Product Identifier', 'MSE_d', 'RMSE_d', 'MAE_d'])
for col in sales_train:
    double_model = ExponentialSmoothing(sales_train[col], trend='add').fit()
    doublemodel_preds = double_model.forecast(6).rename('DES Forecast')
    
    
    mse=mean_squared_error(doublemodel_preds,sales_test[col])
    rmse=np.sqrt(mse)
    mae=mean_absolute_error(doublemodel_preds,sales_test[col])
    
    des_data = des_data.append({'Product Identifier': col, 'MSE_d': mse, 'RMSE_d': rmse, 'MAE_d': mae},ignore_index=True)


# In[228]:


des_data


# ## ==================================================

# # Category Wise Modelling

# # Splitting Dataset

# In[229]:


drinks_and_food_x = drinks_and_food_en.drop(['sales','sell_price','week_id','year','month','day'],axis=1)
fast_moving_consumer_goods_x = fast_moving_consumer_goods_en.drop(['sales','sell_price','week_id','year','month','day'],axis=1)
others_x=others_en.drop(['sales','sell_price','week_id','year','month','day'],axis=1)


# In[230]:


drinks_and_food_y = drinks_and_food_en['sales']
fast_moving_consumer_goods_y = fast_moving_consumer_goods_en['sales']
others_y=others_en['sales']


# In[231]:


x_train_df = drinks_and_food_x.iloc[:107440]
x_test_df =  drinks_and_food_x.iloc[107440:]
y_train_df = drinks_and_food_y.iloc[:107440]
y_test_df =  drinks_and_food_y.iloc[107440:]


# In[232]:


x_train_fmcg = fast_moving_consumer_goods_x.iloc[:183280]
x_test_fmcg =  fast_moving_consumer_goods_x.iloc[183280:]
y_train_fmcg = fast_moving_consumer_goods_y.iloc[:183280]
y_test_fmcg =  fast_moving_consumer_goods_y.iloc[183280:]


# In[233]:


x_train_others = others_x.iloc[:25280]
x_test_others =  others_x.iloc[25280:]
y_train_others = others_y.iloc[:25280]
y_test_others =  others_y.iloc[25280:]


# # --------------------------------------------------------------------------------

# # Linear regression

# ## Drinks and food

# In[234]:


from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(x_train_df, y_train_df)

lr_preds1 = lr_model.predict(x_test_df)


# In[235]:


lr_errors_df = pd.DataFrame(index=y_test_df.index)
lr_errors_df['Modelname'] = 'Linear Regression'
lr_errors_df['Actual'] = y_test_df
lr_errors_df['Predicted'] = lr_preds1
lr_errors_df['Error'] = lr_preds1 - y_test_df
lr_errors_df.head()


# In[236]:


# Calculate the mean absolute error (MAE)
mae_lr_df = mean_absolute_error(y_test_df, lr_preds1)

# Calculate the mean squared error (MSE)
mse_lr_df = mean_squared_error(y_test_df, lr_preds1)
# Calculate the root mean squared error (RMSE)
rmse_lr_df = np.sqrt(mse_lr_df)


# In[237]:


y_train_df1 = y_train_df.resample('M').mean()

lr_errors_df1 = lr_errors_df.resample('M').mean()

# Convert the index to datetime
y_test_df1=y_test_df.copy()
y_test_df1.index = pd.to_datetime(y_test_df1.index)

# Resample the Series based on month
y_test_df1 = y_test_df1.resample('M').mean()


# In[238]:


# Evaluate predictions for Linear Regression
fig = plt.figure(figsize=(14,7))
plt.plot(y_train_df1.index, y_train_df1, label='Train',linewidth=3)
plt.plot(y_test_df1.index, y_test_df1, label='Test',linewidth=3)
plt.plot(lr_errors_df1.index, lr_errors_df1['Predicted'], label='Forecast - Linear Regression',linewidth=3)
plt.legend(loc='best')
plt.xlabel('StartDate')
plt.ylabel('Sales')
plt.title('Forecast using Linear Regression')
plt.show()


# In[239]:


fig = plt.figure(figsize=(14,7))
plt.plot(lr_errors_df1.index, lr_errors_df1.Error, label='Error',linewidth=3)
plt.plot(lr_errors_df1.index, lr_errors_df1.Actual, label='Actual Sales',linewidth=3)
plt.plot(lr_errors_df1.index, lr_errors_df1.Predicted, label='Forecasted-Sales',linewidth=3)
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Linear Regression Forecasting with Actual sales vs errors')
plt.show()


# ## Fast Moving Consumer Goods

# In[240]:


from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(x_train_fmcg, y_train_fmcg)

lr_preds2 = lr_model.predict(x_test_fmcg)


# In[241]:


lr_errors_fmcg = pd.DataFrame(index=y_test_fmcg.index)
lr_errors_fmcg['Modelname'] = 'Linear Regression'
lr_errors_fmcg['Actual'] = y_test_fmcg
lr_errors_fmcg['Predicted'] = lr_preds2
lr_errors_fmcg['Error'] = lr_preds2 - y_test_fmcg
lr_errors_fmcg.head()


# In[242]:


# Calculate the mean absolute error (MAE)
mae_lr_fmcg = mean_absolute_error(y_test_fmcg, lr_preds2)

# Calculate the mean squared error (MSE)
mse_lr_fmcg = mean_squared_error(y_test_fmcg, lr_preds2)
# Calculate the root mean squared error (RMSE)
rmse_lr_fmcg = np.sqrt(mse_lr_fmcg)


# ## Plotting

# In[243]:


y_train_fmcg1 = y_train_fmcg.resample('M').mean()

lr_errors_fmcg1 = lr_errors_fmcg.resample('M').mean()

# Convert the index to datetime
y_test_fmcg1=y_test_fmcg.copy()
y_test_fmcg1.index = pd.to_datetime(y_test_fmcg1.index)

# Resample the Series based on month
y_test_fmcg1 = y_test_fmcg1.resample('M').mean()


# In[244]:


# Evaluate predictions for Linear Regression
fig = plt.figure(figsize=(14,7))
plt.plot(y_train_fmcg1.index, y_train_fmcg1, label='Train',linewidth=3)
plt.plot(y_test_fmcg1.index, y_test_fmcg1, label='Test',linewidth=3)
plt.plot(lr_errors_fmcg1.index, lr_errors_fmcg1['Predicted'], label='Forecast - Linear Regression',linewidth=3)
plt.legend(loc='best')
plt.xlabel('StartDate')
plt.ylabel('Sales')
plt.title('Forecast using Linear Regression')
plt.show()


# In[245]:


fig = plt.figure(figsize=(14,7))
plt.plot(lr_errors_fmcg1.index, lr_errors_fmcg1.Error, label='Error',linewidth=3)
plt.plot(lr_errors_fmcg1.index, lr_errors_fmcg1.Actual, label='Actual Sales',linewidth=3)
plt.plot(lr_errors_fmcg1.index, lr_errors_fmcg1.Predicted, label='Forecasted-Sales',linewidth=3)
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Linear Regression Forecasting with Actual sales vs errors')
plt.show()


# # Others

# In[246]:


from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(x_train_others, y_train_others)

lr_preds3 = lr_model.predict(x_test_others)


# In[247]:


lr_errors_others = pd.DataFrame(index=y_test_others.index)
lr_errors_others['Modelname'] = 'Linear Regression'
lr_errors_others['Actual'] = y_test_others
lr_errors_others['Predicted'] = lr_preds3
lr_errors_others['Error'] = lr_preds3 - y_test_others
lr_errors_others.head()


# In[248]:


# Calculate the mean absolute error (MAE)
mae_lr_others = mean_absolute_error(y_test_others, lr_preds3)

# Calculate the mean squared error (MSE)
mse_lr_others = mean_squared_error(y_test_others, lr_preds3)

# Calculate the root mean squared error (RMSE)
rmse_lr_others = np.sqrt(mse_lr_others)


# ## Plotting

# In[249]:


y_train_others1 = y_train_others.resample('M').mean()

lr_errors_others1 = lr_errors_others.resample('M').mean()

# Convert the index to datetime
y_test_others1=y_test_others.copy()
y_test_others1.index = pd.to_datetime(y_test_others1.index)

# Resample the Series based on month
y_test_others1 = y_test_others1.resample('M').mean()


# In[250]:


# Evaluate predictions for Linear Regression
fig = plt.figure(figsize=(14,7))
plt.plot(y_train_others1.index, y_train_others1, label='Train',linewidth=3)
plt.plot(y_test_others1.index, y_test_others1, label='Test',linewidth=3)
plt.plot(lr_errors_others1.index, lr_errors_others1['Predicted'], label='Forecast - Linear Regression',linewidth=3)
plt.legend(loc='best')
plt.xlabel('StartDate')
plt.ylabel('Sales')
plt.title('Forecast using Linear Regression')
plt.show()


# In[251]:


fig = plt.figure(figsize=(14,7))
plt.plot(lr_errors_others1.index, lr_errors_others1.Error, label='Error',linewidth=3)
plt.plot(lr_errors_others1.index, lr_errors_others1.Actual, label='Actual Sales',linewidth=3)
plt.plot(lr_errors_others1.index, lr_errors_others1.Predicted, label='Forecasted-Sales',linewidth=3)
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Linear Regression Forecasting with Actual sales vs errors')
plt.show()


# # ---------------------------------------------------------------------------

# # Extra Tree Regressor

# In[252]:


from sklearn.ensemble import ExtraTreesRegressor


# ## Drinks and Food

# In[253]:


# fit model
etr_model = ExtraTreesRegressor(n_estimators=100)
etr_model.fit(x_train_df, y_train_df)

etr_preds1 = etr_model.predict(x_test_df)
print('Prediction is done..')


# In[254]:


etr_errors_df = pd.DataFrame(index=y_test_df.index)
etr_errors_df['Modelname'] = 'Extra Trees Regressor'
etr_errors_df['Actual'] = y_test_df
etr_errors_df['Predicted'] = etr_preds1
etr_errors_df['Error'] = etr_preds1 - y_test_df
etr_errors_df.head()


# In[255]:


# Calculate the mean absolute error (MAE)
mae_etr_df = mean_absolute_error(y_test_df, etr_preds1)

# Calculate the mean squared error (MSE)
mse_etr_df = mean_squared_error(y_test_df, etr_preds1)
# Calculate the root mean squared error (RMSE)
rmse_etr_df = np.sqrt(mse_etr_df)


# In[256]:


y_train_df2 = y_train_df.resample('M').mean()

etr_errors_df2 = etr_errors_df.resample('M').mean()

# Convert the index to datetime
y_test_df2=y_test_df.copy()
y_test_df2.index = pd.to_datetime(y_test_df2.index)

# Resample the Series based on month
y_test_df2 = y_test_df2.resample('M').mean()


# In[257]:


# Evaluate predictions for Extra Tree Regressor
fig = plt.figure(figsize=(14,7))
plt.plot(y_train_df2.index, y_train_df2, label='Train',linewidth=3)
plt.plot(y_test_df2.index, y_test_df2, label='Test',linewidth=3)
plt.plot(etr_errors_df2.index, etr_errors_df2['Predicted'], label='Forecast - Extra Tree Regressor',linewidth=3)
plt.legend(loc='best')
plt.xlabel('StartDate')
plt.ylabel('Sales')
plt.title('Forecast using Extra Tree Regressor')
plt.show()


# In[258]:


fig = plt.figure(figsize=(14,7))
plt.plot(etr_errors_df2.index, etr_errors_df2.Error, label='Error',linewidth=3)
plt.plot(etr_errors_df2.index, etr_errors_df2.Actual, label='Actual Sales',linewidth=3)
plt.plot(etr_errors_df2.index, etr_errors_df2.Predicted, label='Forecasted-Sales',linewidth=3)
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Extra Tree Regressor Forecasting with Actual sales vs errors')
plt.show()


# # Fast Moving Consumer Goods

# In[259]:


# fit model
etr_model = ExtraTreesRegressor(n_estimators=100)
etr_model.fit(x_train_fmcg, y_train_fmcg)

etr_preds2 = etr_model.predict(x_test_fmcg)
print('Prediction is done..')


# In[260]:


etr_errors_fmcg = pd.DataFrame(index=y_test_fmcg.index)
etr_errors_fmcg['Modelname'] = 'Extra Trees Regressor'
etr_errors_fmcg['Actual'] = y_test_fmcg
etr_errors_fmcg['Predicted'] = etr_preds2
etr_errors_fmcg['Error'] = etr_preds2 - y_test_fmcg
etr_errors_fmcg.head()


# In[261]:


# Calculate the mean absolute error (MAE)
mae_etr_fmcg = mean_absolute_error(y_test_fmcg, etr_preds2)

# Calculate the mean squared error (MSE)
mse_etr_fmcg = mean_squared_error(y_test_fmcg, etr_preds2)
# Calculate the root mean squared error (RMSE)
rmse_etr_fmcg = np.sqrt(mse_etr_fmcg)


# In[262]:


y_train_fmcg2 = y_train_fmcg.resample('M').mean()

etr_errors_fmcg2 = etr_errors_fmcg.resample('M').mean()

# Convert the index to datetime
y_test_fmcg2=y_test_fmcg.copy()
y_test_fmcg2.index = pd.to_datetime(y_test_fmcg2.index)

# Resample the Series based on month
y_test_fmcg2 = y_test_fmcg2.resample('M').mean()


# In[263]:


# Evaluate predictions for Extra Tree Regressor
fig = plt.figure(figsize=(14,7))
plt.plot(y_train_fmcg2.index, y_train_fmcg2, label='Train',linewidth=3)
plt.plot(y_test_fmcg2.index, y_test_fmcg2, label='Test',linewidth=3)
plt.plot(etr_errors_fmcg2.index, etr_errors_fmcg2['Predicted'], label='Forecast - Extra Tree Regressor',linewidth=3)
plt.legend(loc='best')
plt.xlabel('StartDate')
plt.ylabel('Sales')
plt.title('Forecast using Extra Tree Regressor')
plt.show()


# In[264]:


fig = plt.figure(figsize=(14,7))
plt.plot(etr_errors_fmcg2.index, etr_errors_fmcg2.Error, label='Error',linewidth=3)
plt.plot(etr_errors_fmcg2.index, etr_errors_fmcg2.Actual, label='Actual Sales',linewidth=3)
plt.plot(etr_errors_fmcg2.index, etr_errors_fmcg2.Predicted, label='Forecasted-Sales',linewidth=3)
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Extra Tree Regressor Forecasting with Actual sales vs errors')
plt.show()


# # Others

# In[265]:


# fit model
etr_model = ExtraTreesRegressor(n_estimators=100)
etr_model.fit(x_train_others, y_train_others)

etr_preds3 = etr_model.predict(x_test_others)
print('Prediction is done..')


# In[266]:


etr_errors_others = pd.DataFrame(index=y_test_others.index)
etr_errors_others['Modelname'] = 'Extra Trees Regressor'
etr_errors_others['Actual'] = y_test_others
etr_errors_others['Predicted'] = etr_preds3
etr_errors_others['Error'] = etr_preds3 - y_test_others
etr_errors_others.head()


# In[267]:


# Calculate the mean absolute error (MAE)
mae_etr_others = mean_absolute_error(y_test_others, etr_preds3)

# Calculate the mean squared error (MSE)
mse_etr_others = mean_squared_error(y_test_others, etr_preds3)
# Calculate the root mean squared error (RMSE)
rmse_etr_others = np.sqrt(mse_etr_others)


# In[268]:


y_train_others2 = y_train_others.resample('M').mean()

etr_errors_others2 = etr_errors_others.resample('M').mean()

# Convert the index to datetime
y_test_others2=y_test_others.copy()
y_test_others2.index = pd.to_datetime(y_test_others2.index)

# Resample the Series based on month
y_test_others2 = y_test_others2.resample('M').mean()


# In[269]:


# Evaluate predictions for Extra Tree Regressor
fig = plt.figure(figsize=(14,7))
plt.plot(y_train_others2.index, y_train_others2, label='Train',linewidth=3)
plt.plot(y_test_others2.index, y_test_others2, label='Test',linewidth=3)
plt.plot(etr_errors_others2.index, etr_errors_others2['Predicted'], label='Forecast - Extra Tree Regressor',linewidth=3)
plt.legend(loc='best')
plt.xlabel('StartDate')
plt.ylabel('Sales')
plt.title('Forecast using Extra Tree Regressor')
plt.show()


# In[270]:


fig = plt.figure(figsize=(14,7))
plt.plot(etr_errors_others2.index, etr_errors_others2.Error, label='Error',linewidth=3)
plt.plot(etr_errors_others2.index, etr_errors_others2.Actual, label='Actual Sales',linewidth=3)
plt.plot(etr_errors_others2.index, etr_errors_others2.Predicted, label='Forecasted-Sales',linewidth=3)
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Extra Tree Regressor Forecasting with Actual sales vs errors')
plt.show()


# # --------------------------------------------------------------------------

# # Multiple Linear regression

# In[271]:


import statsmodels.api as sm


# # Drinks and Food

# In[272]:


# Fit the OLS model
ml_model = sm.OLS(y_train_df, x_train_df).fit()

# Make predictions on the test data
ml_preds1 = ml_model.predict(x_test_df)


# In[273]:


ml_errors_df = pd.DataFrame(index=y_test_df.index)
ml_errors_df['Modelname'] = 'Multi Linear Regression'
ml_errors_df['Actual'] = y_test_df
ml_errors_df['Predicted'] = ml_preds1
ml_errors_df['Error'] = ml_preds1 - y_test_df
ml_errors_df.head()


# In[274]:


# Calculate the mean absolute error (MAE)
mae_ml_df = mean_absolute_error(y_test_df, ml_preds1)

# Calculate the mean squared error (MSE)
mse_ml_df = mean_squared_error(y_test_df, ml_preds1)
# Calculate the root mean squared error (RMSE)
rmse_ml_df = np.sqrt(mse_ml_df)


# In[275]:


y_train_df3 = y_train_df.resample('M').mean()

ml_errors_df3 = ml_errors_df.resample('M').mean()

# Convert the index to datetime
y_test_df3=y_test_df.copy()
y_test_df3.index = pd.to_datetime(y_test_df3.index)

# Resample the Series based on month
y_test_df3 = y_test_df3.resample('M').mean()


# In[276]:


# Evaluate predictions for Multiple Linear Regressor
fig = plt.figure(figsize=(14,7))
plt.plot(y_train_df3.index, y_train_df3, label='Train',linewidth=3)
plt.plot(y_test_df3.index, y_test_df3, label='Test',linewidth=3)
plt.plot(ml_errors_df3.index, ml_errors_df3['Predicted'], label='Forecast - Multiple Linear Regressor',linewidth=3)
plt.legend(loc='best')
plt.xlabel('StartDate')
plt.ylabel('Sales')
plt.title('Forecast using Multiple Linear Regression')
plt.show()


# In[277]:


# Evaluate predictions for Linear Regression
fig = plt.figure(figsize=(14,7))
plt.plot(ml_errors_df3.index, ml_errors_df3.Error, label='Error',linewidth=3)
plt.plot(ml_errors_df3.index, ml_errors_df3.Actual, label='Actual Sales',linewidth=3)
plt.plot(ml_errors_df3.index, ml_errors_df3.Predicted, label='Forecasted-Sales',linewidth=3)
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Multiple Linear Regressor Forecasting with Actual sales vs errors')
plt.show()


# # Fast Moving Consumer Goods

# In[278]:


# Fit the OLS model
ml_model = sm.OLS(y_train_fmcg, x_train_fmcg).fit()

# Make predictions on the test data
ml_preds2 = ml_model.predict(x_test_fmcg)


# In[279]:


ml_errors_fmcg = pd.DataFrame(index=y_test_fmcg.index)
ml_errors_fmcg['Modelname'] = 'Multi Linear Regression'
ml_errors_fmcg['Actual'] = y_test_fmcg
ml_errors_fmcg['Predicted'] = ml_preds2
ml_errors_fmcg['Error'] = ml_preds2 - y_test_fmcg
ml_errors_fmcg.head()


# In[280]:


# Calculate the mean absolute error (MAE)
mae_ml_fmcg = mean_absolute_error(y_test_fmcg, ml_preds2)

# Calculate the mean squared error (MSE)
mse_ml_fmcg = mean_squared_error(y_test_fmcg, ml_preds2)
# Calculate the root mean squared error (RMSE)
rmse_ml_fmcg = np.sqrt(mse_ml_fmcg)


# In[281]:


y_train_fmcg3 = y_train_fmcg.resample('M').mean()

ml_errors_fmcg3 = ml_errors_fmcg.resample('M').mean()

# Convert the index to datetime
y_test_fmcg3=y_test_others.copy()
y_test_fmcg3.index = pd.to_datetime(y_test_fmcg3.index)

# Resample the Series based on month
y_test_fmcg3 = y_test_fmcg3.resample('M').mean()


# In[282]:


# Evaluate predictions for Extra Tree Regressor
fig = plt.figure(figsize=(14,7))
plt.plot(y_train_fmcg3.index, y_train_fmcg3, label='Train',linewidth=3)
plt.plot(y_test_fmcg3.index, y_test_fmcg3, label='Test',linewidth=3)
plt.plot(ml_errors_fmcg3.index, ml_errors_fmcg3['Predicted'], label='Forecast - Multiple Linear Regressor',linewidth=3)
plt.legend(loc='best')
plt.xlabel('StartDate')
plt.ylabel('Sales')
plt.title('Forecast using Multiple Linear Regressor')
plt.show()


# In[283]:


# Evaluate predictions for Linear Regression
fig = plt.figure(figsize=(14,7))
plt.plot(ml_errors_fmcg3.index, ml_errors_fmcg3.Error, label='Error',linewidth=3)
plt.plot(ml_errors_fmcg3.index, ml_errors_fmcg3.Actual, label='Actual Sales',linewidth=3)
plt.plot(ml_errors_fmcg3.index, ml_errors_fmcg3.Predicted, label='Forecasted-Sales',linewidth=3)
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Multiple Linear Regressor Forecasting with Actual sales vs errors')
plt.show()


# # Others

# In[284]:


# Fit the OLS model
ml_model = sm.OLS(y_train_others, x_train_others).fit()

# Make predictions on the test data
ml_preds3 = ml_model.predict(x_test_others)


# In[285]:


ml_errors_others = pd.DataFrame(index=y_test_others.index)
ml_errors_others['Modelname'] = 'Multi Linear Regression'
ml_errors_others['Actual'] = y_test_others
ml_errors_others['Predicted'] = ml_preds3
ml_errors_others['Error'] = ml_preds3 - y_test_others
ml_errors_others.head()


# In[286]:


# Calculate the mean absolute error (MAE)
mae_ml_others = mean_absolute_error(y_test_others, ml_preds3)

# Calculate the mean squared error (MSE)
mse_ml_others = mean_squared_error(y_test_others, ml_preds3)
# Calculate the root mean squared error (RMSE)
rmse_ml_others = np.sqrt(mse_ml_others)


# In[287]:


y_train_others3 = y_train_others.resample('M').mean()

ml_errors_others3 = ml_errors_others.resample('M').mean()

# Convert the index to datetime
y_test_others3=y_test_others.copy()
y_test_others3.index = pd.to_datetime(y_test_others3.index)

# Resample the Series based on month
y_test_others3 = y_test_others3.resample('M').mean()


# In[288]:


# Evaluate predictions for Extra Tree Regressor
fig = plt.figure(figsize=(14,7))
plt.plot(y_train_others3.index, y_train_others3, label='Train',linewidth=3)
plt.plot(y_test_others3.index, y_test_others3, label='Test',linewidth=3)
plt.plot(ml_errors_others3.index, ml_errors_others3['Predicted'], label='Forecast - Multiple Linear Regressor',linewidth=3)
plt.legend(loc='best')
plt.xlabel('StartDate')
plt.ylabel('Sales')
plt.title('Forecast using Multiple Linear Regressor')
plt.show()


# In[289]:


# Evaluate predictions for Linear Regression
fig = plt.figure(figsize=(14,7))
plt.plot(ml_errors_others3.index, ml_errors_others3.Error, label='Error',linewidth=3)
plt.plot(ml_errors_others3.index, ml_errors_others3.Actual, label='Actual Sales',linewidth=3)
plt.plot(ml_errors_others3.index, ml_errors_others3.Predicted, label='Forecasted-Sales',linewidth=3)
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Multiple Linear Regressor Forecasting with Actual sales vs errors')
plt.show()


# # -----------------------------------------------------------------------------------------

# ## Evaluation Metrics (MAE/RMSE/MAE)

# ### Product Wise

# In[312]:


evaluation_data = SA_data.merge(prophet_data, on='Product Identifier', how='inner').merge(des_data, on='Product Identifier', how='inner')

evaluation_data


# ### Category Wise

# #### Drinks and Food

# In[291]:


SA_errors_df = SA_errors_df.groupby('Modelname').agg(
    Total_Sales=('Actual', 'sum'),
    Total_Pred_Sales=('Predicted', 'sum'),
    Model_Overall_Error=('Error', 'sum'),
)

SA_errors_df['MAE'] = mae_sa_df
SA_errors_df['MSE'] = mse_sa_df
SA_errors_df['RMSE'] = rmse_sa_df


# In[292]:


result_lr_df = lr_errors_df.groupby('Modelname').agg(
    Total_Sales=('Actual', 'sum'),
    Total_Pred_Sales=('Predicted', 'sum'),
    Model_Overall_Error=('Error', 'sum'),
)

result_lr_df['MAE'] = mae_lr_df
result_lr_df['MSE'] = mse_lr_df
result_lr_df['RMSE'] = rmse_lr_df


# In[294]:


result_etr_df = etr_errors_df.groupby('Modelname').agg(
    Total_Sales=('Actual', 'sum'),
    Total_Pred_Sales=('Predicted', 'sum'),
    Model_Overall_Error=('Error', 'sum'),
)

result_etr_df['MAE'] = mae_etr_df
result_etr_df['MSE'] = mse_etr_df
result_etr_df['RMSE'] = rmse_etr_df


# In[295]:


result_ml_df = ml_errors_df.groupby('Modelname').agg(
    Total_Sales=('Actual', 'sum'),
    Total_Pred_Sales=('Predicted', 'sum'),
    Model_Overall_Error=('Error', 'sum'),
)

result_ml_df['MAE'] = mae_ml_df
result_ml_df['MSE'] = mse_ml_df
result_ml_df['RMSE'] = rmse_ml_df


# In[296]:


result_pr_df = pr_errors_df.groupby('Modelname').agg(
    Total_Sales=('Actual', 'sum'),
    Total_Pred_Sales=('Predicted', 'sum'),
    Model_Overall_Error=('Error', 'sum'),
)

result_pr_df['MAE'] = mae_pr_df
result_pr_df['MSE'] = mse_pr_df
result_pr_df['RMSE'] = rmse_pr_df


# In[297]:


list_objs = [SA_errors_df,result_pr_df,result_lr_df,result_etr_df,result_ml_df]
metrics_table1 = pd.concat(list_objs)
metrics_table1


# #### Fast Moving Consumer Goods

# In[313]:


SA_errors_fmcg = SA_errors_fmcg.groupby('Modelname').agg(
    Total_Sales=('Actual', 'sum'),
    Total_Pred_Sales=('Predicted', 'sum'),
    Model_Overall_Error=('Error', 'sum'),
)

SA_errors_fmcg['MAE'] = mae_sa_fmcg
SA_errors_fmcg['MSE'] = mse_sa_fmcg
SA_errors_fmcg['RMSE'] = rmse_sa_fmcg


# In[300]:


result_lr_fmcg = lr_errors_fmcg.groupby('Modelname').agg(
    Total_Sales=('Actual', 'sum'),
    Total_Pred_Sales=('Predicted', 'sum'),
    Model_Overall_Error=('Error', 'sum'),
)

result_lr_fmcg['MAE'] = mae_lr_fmcg
result_lr_fmcg['MSE'] = mse_lr_fmcg
result_lr_fmcg['RMSE'] = rmse_lr_fmcg


# In[301]:


result_etr_fmcg = etr_errors_fmcg.groupby('Modelname').agg(
    Total_Sales=('Actual', 'sum'),
    Total_Pred_Sales=('Predicted', 'sum'),
    Model_Overall_Error=('Error', 'sum'),
)

result_etr_fmcg['MAE'] = mae_etr_fmcg
result_etr_fmcg['MSE'] = mse_etr_fmcg
result_etr_fmcg['RMSE'] = rmse_etr_fmcg


# In[302]:


result_ml_fmcg = ml_errors_fmcg.groupby('Modelname').agg(
    Total_Sales=('Actual', 'sum'),
    Total_Pred_Sales=('Predicted', 'sum'),
    Model_Overall_Error=('Error', 'sum'),
)

result_ml_fmcg['MAE'] = mae_ml_fmcg
result_ml_fmcg['MSE'] = mse_ml_fmcg
result_ml_fmcg['RMSE'] = rmse_ml_fmcg


# In[303]:


result_pr_fmcg = pr_errors_fmcg.groupby('Modelname').agg(
    Total_Sales=('Actual', 'sum'),
    Total_Pred_Sales=('Predicted', 'sum'),
    Model_Overall_Error=('Error', 'sum'),
)

result_pr_fmcg['MAE'] = mae_pr_fmcg
result_pr_fmcg['MSE'] = mse_pr_fmcg
result_pr_fmcg['RMSE'] = rmse_pr_fmcg


# In[304]:


list_objs = [SA_errors_fmcg,result_pr_fmcg,result_lr_fmcg,result_etr_fmcg,result_ml_fmcg]
metrics_table2 = pd.concat(list_objs)
metrics_table2


# #### Others

# In[305]:


SA_errors_others = SA_errors_others.groupby('Modelname').agg(
    Total_Sales=('Actual', 'sum'),
    Total_Pred_Sales=('Predicted', 'sum'),
    Model_Overall_Error=('Error', 'sum'),
)

SA_errors_others['MAE'] = mae_sa_others
SA_errors_others['MSE'] = mse_sa_others
SA_errors_others['RMSE'] = rmse_sa_others


# In[306]:


result_lr_others = lr_errors_others.groupby('Modelname').agg(
    Total_Sales=('Actual', 'sum'),
    Total_Pred_Sales=('Predicted', 'sum'),
    Model_Overall_Error=('Error', 'sum'),
)

result_lr_others['MAE'] = mae_lr_others
result_lr_others['MSE'] = mse_lr_others
result_lr_others['RMSE'] = rmse_lr_others


# In[307]:


result_etr_others = etr_errors_others.groupby('Modelname').agg(
    Total_Sales=('Actual', 'sum'),
    Total_Pred_Sales=('Predicted', 'sum'),
    Model_Overall_Error=('Error', 'sum'),
)

result_etr_others['MAE'] = mae_etr_others
result_etr_others['MSE'] = mse_etr_others
result_etr_others['RMSE'] = rmse_etr_others


# In[308]:


result_ml_others = ml_errors_others.groupby('Modelname').agg(
    Total_Sales=('Actual', 'sum'),
    Total_Pred_Sales=('Predicted', 'sum'),
    Model_Overall_Error=('Error', 'sum'),
)

result_ml_others['MAE'] = mae_ml_others
result_ml_others['MSE'] = mse_ml_others
result_ml_others['RMSE'] = rmse_ml_others


# In[309]:


result_pr_others = pr_errors_others.groupby('Modelname').agg(
    Total_Sales=('Actual', 'sum'),
    Total_Pred_Sales=('Predicted', 'sum'),
    Model_Overall_Error=('Error', 'sum'),
)

result_pr_others['MAE'] = mae_pr_others
result_pr_others['MSE'] = mse_pr_others
result_pr_others['RMSE'] = rmse_pr_others


# In[311]:


list_objs = [SA_errors_others,result_pr_others,result_lr_others,result_etr_others,result_ml_others]
metrics_table2 = pd.concat(list_objs)
metrics_table2


# # ------------------------------------------------------------------------------------
