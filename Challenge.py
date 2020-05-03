# Import dependencies into Jupyter Notebook
import json
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import config
import time

# Remove alternate titles to come up with a clean movie with one title per movie
def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    # combine alternate titles into one list
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune-Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles

    # Change column names to more descriptive names
    def change_column_name(old_name, new_name):
        if old_name in movie:
            movie[new_name] = movie.pop(old_name)
    change_column_name('Adaptation by', 'Writer(s)')
    change_column_name('Country of origin', 'Country')
    change_column_name('Directed by', 'Director')
    change_column_name('Distributed by', 'Distributor')
    change_column_name('Edited by', 'Editor(s)')
    change_column_name('Length', 'Running time')
    change_column_name('Original release', 'Release date')
    change_column_name('Music by', 'Composer(s)')
    change_column_name('Produced by', 'Producer(s)')
    change_column_name('Producer', 'Producer(s)')
    change_column_name('Productioncompanies ', 'Production company(s)')
    change_column_name('Productioncompany ', 'Production company(s)')
    change_column_name('Released', 'Release Date')
    change_column_name('Release Date', 'Release date')
    change_column_name('Screen story by', 'Writer(s)')
    change_column_name('Screenplay by', 'Writer(s)')
    change_column_name('Story by', 'Writer(s)')
    change_column_name('Theme music composer', 'Composer(s)')
    change_column_name('Written by', 'Writer(s)')

    return movie

# Formatting the dollar amounts
def parse_dollars(s):
    # if s is not a string, return NaN
    if type(s) != str:
        return np.nan

    # if input is of the form $###.# million
    if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " million"
        s = re.sub(r'\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a million
        value = float(s) * 10**6

        # return value
        return value

    # if input is of the form $###.# billion
    elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " billion"
        s = re.sub(r'\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a billion
        value = float(s) * 10**9

        # return value
        return value

    # if input is of the form $###,###,###
    elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

        # remove dollar sign and commas
        s = re.sub(r'\$|,','', s)

        # convert to float
        value = float(s)

        # return value
        return value

    # otherwise, return NaN
    else:
        return np.nan 


# Loading json and csv files
def loading_files(wikipedia, movies_metadata, movie_ratings):
    with open(wikipedia, mode='r') as file:
        wiki_movies_raw = json.load(file)
    kaggle_metadata = pd.read_csv(movies_metadata, low_memory=False)
    ratings = pd.read_csv(movie_ratings)

    # Create wiki_movies_raw DataFrame to work with data
    wiki_movies_raw_df = pd.DataFrame(wiki_movies_raw)
    wiki_movies_raw_df.head()
    wiki_movies_raw_df.columns.tolist()

    # Defining subset of movies that has a director and imbd link as wiki_movies
    wiki_movies = [movie for movie in wiki_movies_raw
                if ('Director' in movie or 'Directed by' in movie)
                    and 'imdb_link' in movie
                    and 'No. of episodes' not in movie]
    
    # Creating dataframe from wiki_movies variable
    wiki_movies_df = pd.DataFrame(wiki_movies)

    # Call clean movies
    clean_movies = [clean_movie(movie) for movie in wiki_movies]

    #Creating clean movies dataframe
    wiki_movies_df = pd.DataFrame(clean_movies)
    
    # Creating additional column imbd_ID in wiki_movies dataframe and dropping duplicates
    wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
    wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)

    #Getting rid of columns with more than 90% Null values in wiki_movies dataframe
    wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
    wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]

    #Dropping box office null values
    box_office = wiki_movies_df['Box office'].dropna() 
    
    # Creating a list of box office dollar amounts
    lambda x: type(x) != str
    box_office[box_office.map(lambda x: type(x) != str)]
    box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)

    # Create two Series for box office, to be able to then match them
    form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
    #box_office.str.contains(form_one, flags=re.IGNORECASE).sum()
    form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'
    #box_office.str.contains(form_two, flags=re.IGNORECASE).sum()
    matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)
    matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)
    
    # Cleaning amounts so that figures will only be in digits, not words, and prepend with dollar sign
    box_office[~matches_form_one & ~matches_form_two]
    box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

    # Match almost all the box office values, we’ll use these values to extract only 
    # the parts of the strings that match returns a DataFrame where every column is the data that matches a capture group
    box_office.str.extract(f'({form_one}|{form_two})')

    # Extract box office data from wiki movies dataframe
    wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
    wiki_movies_df.drop('Box office', axis=1, inplace=True)

    # Transform budget to extract all data from Budget to put into budget (non capitalized)
    budget = wiki_movies_df['Budget'].dropna()
    budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)
    budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
    matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
    matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)
    budget[~matches_form_one & ~matches_form_two]
    budget = budget.str.replace(r'\[\d+\]\s*', '')
    budget[~matches_form_one & ~matches_form_two]

    # Line up two budget columns from merged dataframe
    wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

    #Dropping extra Budget column from wiki_movies dataframe
    wiki_movies_df.drop('Budget', axis=1, inplace=True)

    # Concatonating elements of Release Date column to make them into a list
    release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

    # Defining formatting of dates
    date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
    date_form_two = r'\d{4}.[01]\d.[123]\d'
    date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
    date_form_four = r'\d{4}'

    # Create dataframe organizing movies by release date to convert into one format
    wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)

    # Create Running Time variable make a variable that holds the non-null values of Release date in the DataFrame, converting lists to string
    running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

    # Extract running time data strings
    running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')

    # Change running time from string to numeric
    running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)

    # Convert the hour capture groups and minute capture groups to minutes if the pure minutes capture group is zero
    wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)

    # Drop running time from dataset
    wiki_movies_df.drop('Running time', axis=1, inplace=True)

    # Remove bad data in Adult category
    kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]

    # Keep rows where the adult column is False, and then drop the adult column
    kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')

    # Converting Budget, id and Popularity columns to int, numeric and numeric, respectively
    kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
    kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
    kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')

    # Convert Release Date to DateTime format
    kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])

    # Convert timestamp to datetime data type
    pd.to_datetime(ratings['timestamp'], unit='s')

    # Assign to Timestamp Column
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

    # Inner join of movies_df and Kaggle Metadata
    movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])

    # Documenting what we are doing with Competing data:
    # Wiki (json)        Movielens (2 CSVs: ratings and movies)       Resolution
    #--------------------------------------------------------------------------
    # title_wiki               title_kaggle                          Drop Wikipedia
    # running_time             runtime                               Keep Kaggle; fill in zeros with Wikipedia data
    # budget_wiki              budget_kaggle                         Keep Kaggle; fill in zeros with Wikipedia data
    # box_office               revenue                               Keep Kaggle; fill in zeros with Wikipedia data
    # release_date_wiki        release_date_kaggle                   Drop Wikipedia
    # Language                 original_language                     Drop Wikipedia
    # Production company(s)    production_companies                  Drop Wikipedia


    # Dropping from the DataFrame an incorrectly merged movies: 
    # The Holiday in the Wikipedia data got merged with From Here to Eternity
    movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)

    # Convert the lists in Language to tuples so that the value_counts() method will work
    movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)

    # First, drop the title_wiki, release_date_wiki, Language, and Production company(s) columns.
    movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)

    # Make a function that fills in missing data for a column pair and then drops the redundant column
    def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
        df[kaggle_column] = df.apply(
            lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
            , axis=1)
        df.drop(columns=wiki_column, inplace=True)

    # Run the function for the three column pairs that we decided to fill in zeros
    fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
    fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
    fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')
    
    # Remove Video Column
    movies_df['video'].value_counts(dropna=False)

    # Reordering Columns in groupings: Identifying information (IDs, titles, URLs, etc.)
    # Quantitative facts (runtime, budget, revenue, etc.)
    # Qualitative facts (genres, languages, country, etc.)
    # Business data (production companies, distributors, etc.)
    # People (producers, director, cast, writers, etc.)
    movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                        'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                        'genres','original_language','overview','spoken_languages','Country',
                        'production_companies','production_countries','Distributor',
                        'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                        ]]

    # Rename columns for consistency
    movies_df.rename({'id':'kaggle_id',
                    'title_kaggle':'title',
                    'url':'wikipedia_url',
                    'budget_kaggle':'budget',
                    'release_date_kaggle':'release_date',
                    'Country':'country',
                    'Distributor':'distributor',
                    'Producer(s)':'producers',
                    'Director':'director',
                    'Starring':'starring',
                    'Cinematography':'cinematography',
                    'Editor(s)':'editors',
                    'Writer(s)':'writers',
                    'Composer(s)':'composers',
                    'Based on':'based_on'
                    }, axis='columns', inplace=True)

    # Pivot this data so that movieId is the index, the columns will be all the rating values, 
    # and the rows will be the counts for each rating value
    rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count() \
                .rename({'userId':'count'}, axis=1) \
                .pivot(index='movieId',columns='rating', values='count')

    # Rename columns so they’re easier to understand by Prepend rating_ to each column with a list comprehension
    rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]

    # Use left merge to keep all movie data and add ratings for movies that have them
    movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')

    # Give movies with no ratings a rating of zero
    movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)

    # Connecting Jupyter to Postgres
    db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/movie_data"
    try:
        engine = create_engine(db_string)
    except ConnectionRefusedError:
        print('Connection Refused')

    # load movies into SQL table
    movies_df.to_sql(name='movies', con=engine, if_exists='replace') 

#Create password
db_password = config.db_password
# Create directory path
file_dir = 'C:/Users/esobieski/Documents/Berkeley/Movies-ETL/'

# Creating inputs for function call
wikipedia = file_dir+'wikipedia.movies.json'
movies_metadata = file_dir+'movies_metadata.csv'
movie_ratings = file_dir+'ratings.csv'
loading_files(wikipedia, movies_metadata, movie_ratings)





