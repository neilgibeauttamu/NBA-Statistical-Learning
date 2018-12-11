Description of Structure:

data:
    expert_ranks - contains expert ranking data collected manually,
                    rank values are the median rank from our aggregation.

    outputs - contains all intermediate and output of data processing scripts.
                (this one could use some cleaning)

    raw_stats - contains output of nba_scrape.py, which is raw stats from nba.com

    reddit_ranks - contains power rankings from nba subreddit (unfortunately was not able
                    to utlize these due to time contraint)

notebooks:
    predictions notebooks - used to perform final predictions on different seasons

    cross_validation_tuning - a large amount of our methods of feature selection and hyperparameter
                                tuning using cross validation live in this notebook.

    days_since_last_played - data preprocessing to obtain day since last played feature done hyperparameter

    nba_MLP - Kevins neural net notebook

scripts:
    nba_scrape - web scraping script to retrieve raw stats form nba.com

    nba_data_process.py - a large amount of data proccessing was done in this script. Basically contains code to 
                            retrieve all features except for days since last played. 
