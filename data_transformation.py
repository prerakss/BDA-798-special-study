import pandas as pd


#----------------------------------------------------------------------------------------------------------
# 1. LOADING DATASETS

# MOVIE LENS DATASETS
ratings_df = pd.read_csv('/Users/prerak/Documents/BDA Special Study/ratings.csv')
movies_df = pd.read_csv('/Users/prerak/Documents/BDA Special Study/movies.csv')
tags_df = pd.read_csv('/Users/prerak/Documents/BDA Special Study/tags.csv')
genome_scores_df = pd.read_csv('/Users/prerak/Documents/BDA Special Study/genome-scores.csv')
genome_tags_df = pd.read_csv('/Users/prerak/Documents/BDA Special Study/genome-tags.csv')
links_df = pd.read_csv('/Users/prerak/Documents/BDA Special Study/links.csv')


# FILE CONTAINING BOX-OFFICE COLLECTION FOR MOVIES
boxoffice_df = pd.read_csv('/Users/prerak/Documents/BDA Special Study/boxoffice.csv')

#----------------------------------------------------------------------------------------------------------
# 2. BUILDING LOGIC TO MERGE "boxoffice_df" with MovieLens datasets

# FINDING AVERAGE RATING FOR EVERY MOVIE (grouped by movie_id)
movie_ratings = ratings_df.groupby('movieId', as_index=False).agg({'userId' : 'count', 'rating' : 'sum'})

# NEW COLUMN WITH AVERAGE RATING
movie_ratings['avg_rating'] = movie_ratings['rating']/movie_ratings['userId']

# MOVIES WITH LESS THAN 25 USER RATINGS ARE REMOVED
movie_ratings_over_25 = movie_ratings.loc[movie_ratings['userId'] >= 25]

# JOINING WITH "movies_df" (to get movie names which will be the primary key now)
df = pd.merge(movie_ratings_over_25,movies_df,how='inner',left_on='movieId',right_on='movieId')

# Some transformations for the final join (2)
df['year'] = df['title'].str[-5:-1]
df['title'] = df['title'].str[:-7]
df['title'] = df['title'].str.strip()
df['year'] = df['year'].str.strip()

boxoffice_df['title'] = boxoffice_df['title'].str.strip()
boxoffice_df['year'] = boxoffice_df['year'].astype('string')
boxoffice_df['year'] = boxoffice_df['year'].str.strip()

# THE FINAL JOIN LOGIC between boxoffice_df (which contains 'lifetime_gross', the response variable) and MovieLens datasets (containing predictors)
df_temp = pd.merge(df,boxoffice_df,how='inner',left_on=['title','year'],right_on=['title','year'])

