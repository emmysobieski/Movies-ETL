# Movies-ETL

# Challenge

# Assumptions made when building the script:

(1) For the main function of removing alternative titles to only have one set of titles, ie removing titles that are in other languages, it assumes that there will be a title left, ie assumes there is a title in English, and assumes that no additional languages will be added in the future

(2) Changing column names assumes original column names stay the stable within the Wiki JSON data

(3) Formatting dollar amounts assumes no new formats coming from inputs in the future

(4) Assume Kaggle metadata stays better than Wiki movie data, as this script prioritizes kaggle data and then fills in blanks with Wiki data:

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


(5) Remove video column action assumes that video will not increase in importance in the future, which is likely a good assumption, as streaming is likely to continue to take share over video releases.

(6) Box office amounts: assume only spelling error is in form one misspelling of Millions (allows for misspelling of billions too) but only for one letter in the word.  Assumes no additional misspellings of any other letter in the word(s).

(7) This script dropped 2 incorrectly merged movies (The Holiday and From Here to Eternity), but this assumes no others in the future will have been merged incorrectly between the Wiki and Kaggle movie data.  It also assumes that this problem does not become more widespread going forward.

**I assume since you are testing my own postgress to see if the tables are there, that you need the config.py to be loaded into Github so I set the password to something I do not use anywhere else, and put it in config.py and did not gitignore it.

# Special notes:

**I used try-except in the loading of data into SQL so that it lets you know if there is an error loading data.  

 db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/movie_data"
 
    try:
    
        engine = create_engine(db_string)
        
    except ConnectionRefusedError:
    
        print('Connection Refused')

# I also have the new data overwrite old data using if_exists=replace to avoid error of having old data block new data from coming in

# load movies into SQL table
    
    movies_df.to_sql(name='movies', con=engine, if_exists='replace') 
