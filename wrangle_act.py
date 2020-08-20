#!/usr/bin/env python
# coding: utf-8

# # Wrangle & Analyze "WeRateDogs" Data
# 
# In this project, three different datasets (twitter-archive-enhanced, image-prediction, tweet-json-text) are wrangled, cleaned and merged together to obtain valuable insights and visualizations.
# 
# For each dataset, first the data is loaded into a dataframe, keeping the tweet_id as index. After this, the datasets are analyzed in order to list any data quality or tidiness issues. They are solved in the third and last phase, Data cleaning, which involves two steps: Code-to solve the issue and Test-to check whether the issue still persists even after the cleaning.
# 
# Once these phases are completed, all the three datasets are merged and three insights are found out. Visualizations are provided for some these insights.

# # Dataset 1 - Twitter Archive

# > Load Dataset 1:

# In[470]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[471]:


twitter_archive = pd.read_csv("twitter-archive-enhanced.csv")
twitter_archive.head(2)


# In[472]:


twitter_archive.set_index("tweet_id", inplace = True)
twitter_archive.head(2)


# > Asses Dataset 1:

# In[473]:


twitter_archive.shape


# In[474]:


twitter_archive.info()


# In[475]:


twitter_archive.describe()


# In[476]:


#Checking whether index (tweet_id) is unique
twitter_archive.index.is_unique


# In[477]:


#Count of Replies in the dataset
(twitter_archive.in_reply_to_status_id).count()


# In[478]:


#Count of Retweets in the dataset
(twitter_archive.retweeted_status_id).count()


# In[479]:


#Multiple URLs in same cell
(twitter_archive.expanded_urls.str.contains(',', na = False)).sum()


# In[480]:


#These are more suitable to be placed in one column than individual columns

print('Doggo', (twitter_archive['doggo'] == 'doggo').sum() )
print ('Floofer', (twitter_archive['floofer'] == 'floofer').sum() )
print('Pupper', (twitter_archive['pupper'] == 'pupper').sum() )
print('Puppo', (twitter_archive['puppo'] == 'puppo').sum() )
print('None', twitter_archive.query(' doggo == "None" & floofer =="None" & pupper =="None" & puppo =="None" ').count()['name'])


# In[481]:


# Highest frequency of Numerator 
twitter_archive.rating_numerator.value_counts().head()


# In[482]:


# Highest frequency of Denominator
twitter_archive.rating_denominator.value_counts().head()


# In[483]:


# Numerator greater than denominator, need for normalized rating.
twitter_archive[twitter_archive.rating_numerator > twitter_archive.rating_denominator][['text','rating_numerator','rating_denominator']].head()


# In[484]:


# Not all rows have denominator as 10, another reason why a normalized rating is needed.
twitter_archive[twitter_archive.rating_denominator > 10][['text','rating_numerator','rating_denominator']].head()


# In[485]:


# Float numerators incorrectly mentioned.
twitter_archive[twitter_archive['text'].str.contains(r'(\d+\.\d*\/\d+)')][['text', 'rating_numerator', 'rating_denominator']].head(3)


# ## Data Quality & Tidiness Issues in Dataset 1 - Twitter Archive
# 
# 1) Some of the tweets in the dataset are retweets.
# 
# 2) Some of the tweets in the dataset are replies.
# 
# 3) Timestamp should be in Date-Time format.
# 
# 4) Values in name column like 'a', 'the', 'such', etc. which are irrelevant and all have lowercase characters.
# 
# 5) Some tuples contain multiple URLS in the expanded_urls column.
# 
# 6) Fix incorrect ratings and normalize all ratings.
# 
# ## Tidiness Issues in Dataset 1 - Twitter Archive
# 
# 1) To drop unnecessary columns to make dataset tidy.
# 
# 2) Dog stage classification like doggo/ floofer/ pupper/ puppo should be represented in one column instead of four columns.

# > Clean Dataset 1:

# In[486]:


# To create a copy of the dataset
clean_twitter_archive = twitter_archive.copy()


# In[487]:


# Dropping Retweets

#Code
clean_twitter_archive = clean_twitter_archive[clean_twitter_archive['retweeted_status_id'].isnull()]

#Test
print(sum(clean_twitter_archive.retweeted_status_user_id.value_counts()))


# In[488]:


# Dropping Replies

#Code
clean_twitter_archive = clean_twitter_archive[clean_twitter_archive['in_reply_to_status_id'].isnull()]

#Test
print(sum(clean_twitter_archive.in_reply_to_status_id.value_counts()))


# In[489]:


# Fixing timestamp datatype to date-time

#Code
clean_twitter_archive.timestamp = pd.to_datetime(clean_twitter_archive.timestamp)

#Test
clean_twitter_archive.timestamp.dtypes


# In[490]:


# Dropping unwanted columns

#Code
clean_twitter_archive.drop(['in_reply_to_status_id','in_reply_to_user_id','retweeted_status_id',
           'retweeted_status_user_id','retweeted_status_timestamp','source'], axis = 1, inplace = True)

#Test
clean_twitter_archive.columns


# In[491]:


# Dropping values in name column are like 'a', 'the', 'such', etc. which are irrelevant and all have lowercase characters.

sum(clean_twitter_archive.name.str.islower())


# In[492]:


#Code
clean_twitter_archive = clean_twitter_archive[~clean_twitter_archive['name'].str.islower()]

#Test
sum(clean_twitter_archive.name.str.islower())


# In[493]:


# Dog stage category represented in one column
# Reference - Code Snippet provided in Udacity Reviews

# Code
clean_twitter_archive.doggo.replace('None', '', inplace=True)
clean_twitter_archive.floofer.replace('None', '', inplace=True)
clean_twitter_archive.pupper.replace('None', '', inplace=True)
clean_twitter_archive.puppo.replace('None', '', inplace=True)


# In[494]:


clean_twitter_archive['dog_stage'] = clean_twitter_archive.doggo + clean_twitter_archive.floofer + clean_twitter_archive.pupper + clean_twitter_archive.puppo


# In[495]:


clean_twitter_archive.loc[clean_twitter_archive.dog_stage == 'doggopupper', 'dog_stage'] = 'doggo, pupper'
clean_twitter_archive.loc[clean_twitter_archive.dog_stage == 'doggopuppo', 'dog_stage'] = 'doggo, puppo'
clean_twitter_archive.loc[clean_twitter_archive.dog_stage == 'doggofloofer', 'dog_stage'] = 'doggo, floofer'


# In[496]:


clean_twitter_archive.loc[clean_twitter_archive.dog_stage == '', 'dog_stage'] = np.nan

clean_twitter_archive.dog_stage.value_counts()


# In[497]:


clean_twitter_archive.drop(['doggo','floofer','pupper','puppo'], axis = 1, inplace = True)

# Test
clean_twitter_archive.columns


# In[498]:


# Fixing multiple expanded URLs

#Code
for index, column in clean_twitter_archive.iterrows():
    clean_twitter_archive.loc[index, 'expanded_urls'] = 'https://twitter.com/dog_rates/status/' + str(index)

#Test
clean_twitter_archive['expanded_urls'].head()


# In[499]:


# Correctly adding the float numerator values

clean_twitter_archive['rating_numerator'] = clean_twitter_archive['rating_numerator'].astype(float)
clean_twitter_archive['rating_denominator'] = clean_twitter_archive['rating_denominator'].astype(float)


# In[500]:


#Code
import re
fraction_ratings = clean_twitter_archive[clean_twitter_archive['text'].str.contains(r"(\d+\.\d*\/\d+)")].index

for index in fraction_ratings:
    rating = re.search('\d+\.\d+\/\d+', clean_twitter_archive.loc[index,:].text).group(0)
    clean_twitter_archive.at[index,'rating_numerator'], clean_twitter_archive.at[index,'rating_denominator'] = rating.split('/')


# In[501]:


#Test
clean_twitter_archive[clean_twitter_archive['text'].str.contains(r"(\d+\.\d*\/\d+)")][['text', 'rating_numerator', 'rating_denominator']]


# In[502]:


# Normalized rating since in some rows numerators are greater than denominators. 

#Code
clean_twitter_archive['normalized_rating'] = clean_twitter_archive['rating_numerator'] / clean_twitter_archive['rating_denominator']

#Test
clean_twitter_archive.head(2)


# # Dataset 2: Image Predictions

# >Load Dataset 2:

# In[503]:


import requests


# In[504]:


url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'

with open('image-predictions.tsv' , 'wb') as file:
    predictions = requests.get(url)
    file.write(predictions.content)


# In[505]:


img_pred = pd.read_csv('image-predictions.tsv', sep = '\t')
img_pred.head(3)


# In[506]:


img_pred.set_index("tweet_id", inplace = True)
img_pred.head(3)


# > Assess Dataset 2:

# In[507]:


img_pred.shape


# In[508]:


img_pred.info()


# In[509]:


img_pred.describe()


# In[510]:


img_pred.columns


# In[511]:


img_pred[['p1','p2','p3']].head()


# In[512]:


sum(img_pred.jpg_url.duplicated())


# In[513]:


img_pred.sample(5)


# ## Data Quality Issues in img_pred dataframe
# 
# 1) Dog Breed prediction columns (p1,p2,p3) contain a '_' between two words
# 
# 2) Drop Duplicate image URLs (66)
# 
# ## Tidiness Issues in img_pred dataframe
# 
# 1) Confusing column Names

# > Clean Dataset 2:

# In[514]:


clean_img_pred = img_pred.copy()


# In[515]:


# Rename Columns
clean_img_pred.columns


# In[516]:


#Code
clean_img_pred.columns = ['image_url','image_number','1st_prediction','1st_prediction_confidence','1st_prediction_isdog','2nd_prediction','2nd_prediction_confidence','2nd_prediction_isdog','3rd_prediction','3rd_prediction_confidence','3rd_prediction_isdog']

#Test
clean_img_pred.columns


# In[517]:


# Removing underscore from Dog Breed prediction

#Code
dog_preds = ['1st_prediction', '2nd_prediction', '3rd_prediction']

for column in dog_preds:
    clean_img_pred[column] = clean_img_pred[column].str.replace('_', ' ').str.title()


# In[518]:


#Test
clean_img_pred[dog_preds].head(3)


# In[519]:


# Remove duplicate Image URLs
clean_img_pred.image_url.duplicated().sum()


# In[520]:


#Code
clean_img_pred.drop_duplicates(subset=['image_url'], keep='last', inplace=True)

#Test
sum(clean_img_pred['image_url'].duplicated())


# # Dataset 3 - Tweet Json

# > Load Dataset 3:

# In[521]:


import tweepy
import json 
import re


# ### Loading using Twitter API

# In[522]:


# References: http://docs.tweepy.org/en/latest/api.html,
#            https://stackoverflow.com/questions/26075001/error-with-tweepy-oauthhandler

consumer_key = 'EMPTIED'
consumer_secret = 'EMPTIED'
access_token = 'EMPTIED'
access_secret = 'EMPTIED'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


# In[523]:


# Reference: https://realpython.com/python-json/


# **After the above step,firstly, we get all tweet ids from tweet archive dataset**
# 
# twitter_ids = list(twitter_arcive.tweet_id.unique())
# 
# **Secondly, we save the gathered data to a file**
# 
# with open("tweet-json.txt", "w") as file: for ids in twitter_ids: print(f"Gather id: {ids}") try:
#         tweet = api.get_status(ids, tweet_mode = "extended")
#         json.dump(tweet._json, file)
#         file.write('\n')
#     except Exception as e:
#         print(f"Error - id: {ids}" + str(e))
#         
# **Lastly, we open the saved file and load data into a data frame**

# In[524]:


with open('tweet-json.txt') as f:
    tw_json = pd.DataFrame((json.loads(line) for line in f), columns = ['id', 'favorite_count', 'retweet_count'])


# In[525]:


tw_json.head()


# ### Loading dataset using .txt file

# In[526]:


with open('tweet-json.txt') as f:
    tw_json = pd.DataFrame((json.loads(line) for line in f), columns = ['id', 'favorite_count', 'retweet_count'])


# In[527]:


tw_json.columns = ['tweet_id', 'favorites', 'retweets']


# In[528]:


tw_json.set_index('tweet_id', inplace=True)

tw_json.head()


# > Assess Dataset 3:

# In[529]:


tw_json.shape


# In[530]:


tw_json.info()


# In[531]:


tw_json.describe()


# In[532]:


tw_json.index.is_unique


# In[533]:


tw_json.favorites.isnull().sum()


# In[534]:


tw_json.retweets.isnull().sum()


# > Here, there doesn't seem to be any specific data quality or tidiness issues which could be further improved or modified.

# # Merging Datasets :

# In[535]:


from functools import reduce

df = reduce(lambda left, right: pd.merge(left, right, on='tweet_id'), [clean_twitter_archive, clean_img_pred, tw_json])


# In[536]:


df.shape


# In[537]:


df.info()


# In[538]:


df.describe()


# In[539]:


df.head(3)


# In[540]:


df.to_csv('twitter_archive_master.csv')


# # Insights and Visualizations

# In[541]:


# Insight 1: What is the year-wise distribution of tweets posted by the twitter account?

insight1 = df['timestamp'].groupby(df.timestamp.dt.year).agg('count')
insight1


# In[542]:


insight1.plot(kind='pie', y='insight1');
plt.ylabel('');
plt.title('Year wise distribution of number of tweets');


# > Out of the mentioned years (2015, 2016, 2017) in the final dataset, we can observe that the most number of tweets were posted in the year 2016 and least in the year 2017.

# In[543]:


# Insight 2 - How the dogs in the dataset can be categorized based on the dog stage?

insight2 = df.copy()

insight2 = insight2.dog_stage.value_counts()
insight2


# In[544]:


insight2.plot(kind='bar', figsize=(10,6));
plt.xlabel('Dog Stages', size=14);
plt.ylabel('Count', size=14);
plt.title('Distribution of Dog Stages', size=14);


# > Majority of the dogs in the final dataset, belonged to 'Pupper' stage. However, the values for this column are available only for 281 tuples. Therefore, more data is required to give a better distribution. 

# In[545]:


# Insight 3 - What is the comparison between mean favorite tweets with respect to predictions in ‘1st_prediction_isdog’ column?

insight3 = df.copy()

insight3_1 = insight3[insight3['1st_prediction_isdog'] == True]
print('Mean of tweets marked as favorites when the first dog prediction was correct ', insight3_1['favorites'].mean())


# In[546]:


insight3_2 = insight3[insight3['1st_prediction_isdog'] == False]
print('Mean of tweets marked as favorites when the first dog prediction was incorrect ', insight3_2['favorites'].mean())


# > The above results show that the mean favorite tweets in which dogs were correctly predicted are HIGHER in comparison to the ones where dogs were incorrectly predicted.

# # Conclusions:
# 
# Like the above, many other insights could be found out from this dataset wrangled from multiple sources. However, to obtain better, accurate insights, more statistical and categorical data relevant to tweets or dogs would be needed, since during data cleaning phase lots of raw data gets removed due to inconsistencies.

# In[ ]:





# In[ ]:




