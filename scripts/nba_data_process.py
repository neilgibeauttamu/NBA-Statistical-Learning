import pandas as pd 
import pprint as pp
from datetime import datetime
date_format = "%Y-%m-%d"
import time
from multiprocessing.dummy import Pool as ThreadPool 
import numpy as np

# Convert MIN string into int
def ConvertMinutes(x):
    return float(x.split(':')[0])

# Convert GAME_DATE_EST into datetime 
def ConvertGameDate(x):
    return datetime.strptime(x.split('T')[0], date_format)

# Calculate Offensive Rating
def CalcOffRtg(x):
    poss = 0.96*((x['FGA'])+(x['TO'])+0.44*(x['FTA'])-(x['OREB']))
    off_rtg = ((x['PTS'])/poss) * 100
    return off_rtg

# Read in data and do some preproccessing
def ImportData(team_stats_csv, summaries_csv):
    team_stats = pd.read_csv(team_stats_csv)
    summaries = pd.read_csv(summaries_csv)
    game_dates = summaries[['GAME_DATE_EST','HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'GAME_ID']].copy()

    team_stats = pd.merge(team_stats, game_dates, on=['GAME_ID'])
    team_stats = team_stats.drop(columns=['TEAM_ABBREVIATION', 'TEAM_CITY'])

    # Add expert rankings
    if team_stats_csv == '../data/raw_stats/2018-2019_team_stats.csv':
        ranks = pd.read_csv('../data/expert_ranks/expert_ranks_2018-2019_v2.csv')
    elif team_stats_csv == '../data/raw_stats/2017-2018_team_stats.csv':
        ranks = pd.read_csv('../data/expert_ranks/expert_ranks_2017-2018.csv')
    
    
    # elif team_stats_csv == '2016-2017_team_stats.csv':
    #     ranks = pd.read_csv('expert_ranks_2016-2017.csv')
    # elif team_stats_csv == '2015-2016_team_stats.csv':
    #     ranks = pd.read_csv('expert_ranks_2015-2016.csv')
    # elif team_stats_csv == '2014-2015_team_stats.csv':
    #     ranks = pd.read_csv('expert_ranks_2014-2015.csv')

    team_stats = pd.merge(team_stats, ranks, on=['TEAM_NAME'])

    # Change GAME_DATE_EST to datetime object
    team_stats['GAME_DATE_EST'] = team_stats['GAME_DATE_EST'].apply(ConvertGameDate)

    # Calculate Off Rtg
    team_stats['OFF_RTG'] = team_stats.apply(CalcOffRtg, axis=1)

    # Add Home or Away suffixes
    away, home = [x for _, x in team_stats.groupby(team_stats['TEAM_ID'] == team_stats['HOME_TEAM_ID'])]
    home = home.add_suffix('_HOME')
    away = away.add_suffix('_AWAY')

    team_stats = pd.merge(home, away, left_on=['GAME_ID_HOME'], right_on=['GAME_ID_AWAY'])
    team_stats = team_stats.drop(columns=['Unnamed: 0_HOME', 'Unnamed: 0_AWAY'])
    team_stats['MIN'] = team_stats['MIN_HOME'].apply(ConvertMinutes)
    team_stats = team_stats.drop(columns=['MIN_HOME', 'MIN_AWAY'])
    
    # Add binary WIN_HOME field
    team_stats['WIN_HOME'] = 0
    team_stats.loc[team_stats['PTS_HOME'] > team_stats['PTS_AWAY'], 'WIN_HOME'] = 1
    return team_stats

# Find the last N games with a common opponent for each game
def LastNCommonOpponents(n, df):
    print("***** FINDING LAST N COMMON OPPONENTS *****")
    iteration = 0
    tuples = pd.DataFrame()

    for index, i in df.iterrows():
        # Create Lists and Variables
        tuples_a = pd.DataFrame()
        tuples_b = pd.DataFrame()
        team_a_games = []
        team_b_games = []
        team_a_opponents = []
        team_b_opponents = []
        distances = []
        selected_games = []
        team_a = i['TEAM_NAME_HOME']
        team_b = i['TEAM_NAME_AWAY']
        game_date = i['GAME_DATE_EST_HOME']

        # Find common matchups
        for fuck, j in df.iloc[index:].iterrows():
            if (team_a == j['TEAM_NAME_HOME'] or team_a == j['TEAM_NAME_AWAY']) and j['GAME_DATE_EST_HOME'] < game_date:
                if team_a == j['TEAM_NAME_HOME']:
                    team_a_opponents.append(j['TEAM_NAME_AWAY'])
                    j = j.append(pd.Series({'OPPONENT': 'AWAY'}))
                else:
                    team_a_opponents.append(j['TEAM_NAME_HOME'])
                    j = j.append(pd.Series({'OPPONENT': 'HOME'}))
                team_a_games.append(j)
            
            elif (team_b == j['TEAM_NAME_HOME'] or team_b == j['TEAM_NAME_AWAY']) and j['GAME_DATE_EST_HOME'] < game_date:
                if team_b == j['TEAM_NAME_HOME']:
                    team_b_opponents.append(j['TEAM_NAME_AWAY'])
                    j = j.append(pd.Series({'OPPONENT': 'AWAY'}))
                else:
                    team_b_opponents.append(j['TEAM_NAME_HOME'])
                    j = j.append(pd.Series({'OPPONENT': 'HOME'}))
                team_b_games.append(j)

            if len(team_a_games) >= 82 and len(team_b_games) >= 82:
                break

        # Calculate distances 
        for k in range(0, len(team_a_games)):
            for l in range(0, len(team_b_games)):
                    if team_a_opponents[k] == team_b_opponents[l]:
                        distance_a = i['GAME_DATE_EST_HOME'] - team_a_games[k]['GAME_DATE_EST_HOME'] 
                        distance_b = i['GAME_DATE_EST_HOME'] - team_b_games[l]['GAME_DATE_EST_HOME']
                        total_distance = distance_a + distance_b
                        distances.append({
                                            'game_a': team_a_games[k].to_frame().transpose(),
                                            'game_b': team_b_games[l].to_frame().transpose(),
                                            'distance': total_distance
                                        })
        
        sorted_distances = sorted(distances, key=lambda x: x['distance']) 
        temp = (sorted_distances[:n])
        count = 0

        if len(temp) >= n:
            previous_games_list_a = []
            previous_games_list_b = []
            [previous_games_list_a.append(d['game_a']) for d in temp]
            [previous_games_list_b.append(d['game_b']) for d in temp]
            previous_games_a = pd.concat(previous_games_list_a)
            previous_games_b = pd.concat(previous_games_list_b)

            tuples_a = pd.DataFrame()
            first_tuple_a = previous_games_a.iloc[0]
            first_tuple_a = first_tuple_a.add_suffix('_PREV_GAME_A_' + str(0))
            first_tuple_a['CURRENT_GAME_ID'] = i['GAME_ID_HOME']
            first_tuple_a['WIN_HOME'] = i['WIN_HOME']
            first_tuple_a['GAME_DATE_EST'] = i['GAME_DATE_EST_HOME']
            first_tuple_a['TEAM_NAME_HOME'] = i['TEAM_NAME_HOME']
            first_tuple_a['TEAM_NAME_AWAY'] = i['TEAM_NAME_AWAY'] 
            first_tuple_a['EXPERT_RANK_HOME'] = i['EXPERT_RANK_HOME'] 
            first_tuple_a['EXPERT_RANK_AWAY'] = i['EXPERT_RANK_AWAY'] 
            tuples_a = tuples_a.append(first_tuple_a)
            previous_games_a = previous_games_a.iloc[1:]

            for index4, k in previous_games_a.iterrows(): 
                k = k.add_suffix('_PREV_GAME_A_' + str(index4))
                k['CURRENT_GAME_ID'] = i['GAME_ID_HOME'] 
                to_merge = k.to_frame().transpose()
                to_merge['CURRENT_GAME_ID'] = to_merge['CURRENT_GAME_ID'].apply(int)
                tuples_a['CURRENT_GAME_ID'] = tuples_a['CURRENT_GAME_ID'].apply(int)
                tuples_a = pd.merge(tuples_a, to_merge, on=['CURRENT_GAME_ID'], sort=False)
            
            tuples_b = pd.DataFrame()
            first_tuple_b = previous_games_b.iloc[0]
            first_tuple_b = first_tuple_b.add_suffix('_PREV_GAME_B_' + str(0))
            first_tuple_b['CURRENT_GAME_ID'] = i['GAME_ID_HOME'] 
            tuples_b = tuples_b.append(first_tuple_b)
            previous_games_b = previous_games_b.iloc[1:]

            for index5, l in previous_games_b.iterrows(): 
                l = l.add_suffix('_PREV_GAME_B_' + str(index5))
                l['CURRENT_GAME_ID'] = i['GAME_ID_HOME']
                to_merge = l.to_frame().transpose()
                to_merge['CURRENT_GAME_ID'] = to_merge['CURRENT_GAME_ID'].apply(int)
                tuples_b['CURRENT_GAME_ID'] = tuples_b['CURRENT_GAME_ID'].apply(int) 
                tuples_b = pd.merge(tuples_b, to_merge, on=['CURRENT_GAME_ID'], sort=False)

            tuples = tuples.append(pd.merge(tuples_a, tuples_b, on=['CURRENT_GAME_ID'], sort=False))
        
        print ('COMPLETED ITERATION: ', iteration)
        iteration += 1

    return tuples

# Find the last N games each team has played for each game
def LastNGames(n, df):
    print("***** FINDING LAST N GAMES *****")
    iteration = 0
    tuples = pd.DataFrame()

    for index, i in df.iterrows():
        previous_games_a = pd.DataFrame()
        previous_games_b = pd.DataFrame()
        team_a = i['TEAM_NAME_HOME']
        team_b = i['TEAM_NAME_AWAY']
        game_date = i['GAME_DATE_EST_HOME']

        # Get previous games for team a
        for index2, j in df.iloc[index:].iterrows():
            if team_a == j['TEAM_NAME_HOME'] and j['GAME_DATE_EST_HOME'] < game_date:
                    j = j.append(pd.Series({'OPPONENT': 'AWAY'}))
                    previous_games_a = previous_games_a.append(j, ignore_index = True)

            elif team_a == j['TEAM_NAME_AWAY'] and j['GAME_DATE_EST_HOME'] < game_date:
                    j = j.append(pd.Series({'OPPONENT': 'HOME'}))
                    previous_games_a = previous_games_a.append(j, ignore_index = True)

            if len(previous_games_a) >= n:
                break
        
        # Get previous games for team b
        for index3, m in df.iloc[index:].iterrows():
            if team_b == m['TEAM_NAME_HOME'] and m['GAME_DATE_EST_HOME'] < game_date:
                    m = m.append(pd.Series({'OPPONENT': 'AWAY'}))
                    previous_games_b = previous_games_b.append(m, ignore_index = True)
                    
            elif team_b == m['TEAM_NAME_AWAY'] and m['GAME_DATE_EST_HOME'] < game_date:
                    m = m.append(pd.Series({'OPPONENT': 'HOME'}))
                    previous_games_b = previous_games_b.append(m, ignore_index = True)

            if len(previous_games_b) >= n:
                break
            
        if previous_games_a.shape[0] >= n and previous_games_b.shape[0] >= n:
            
            tuples_a = pd.DataFrame()
            first_tuple_a = previous_games_a.iloc[0]
            first_tuple_a = first_tuple_a.add_suffix('_PREV_GAME_A_' + str(0))
            first_tuple_a['CURRENT_GAME_ID'] = i['GAME_ID_HOME']
            first_tuple_a['WIN_HOME'] = i['WIN_HOME']
            first_tuple_a['GAME_DATE_EST'] = i['GAME_DATE_EST_HOME']
            first_tuple_a['TEAM_NAME_HOME'] = i['TEAM_NAME_HOME']
            first_tuple_a['TEAM_NAME_AWAY'] = i['TEAM_NAME_AWAY'] 
            first_tuple_a['EXPERT_RANK_HOME'] = i['EXPERT_RANK_HOME'] 
            first_tuple_a['EXPERT_RANK_AWAY'] = i['EXPERT_RANK_AWAY'] 
            tuples_a = tuples_a.append(first_tuple_a)
            previous_games_a = previous_games_a.iloc[1:]

            for index4, k in previous_games_a.iterrows(): 
                k = k.add_suffix('_PREV_GAME_A_' + str(index4))
                k['CURRENT_GAME_ID'] = i['GAME_ID_HOME'] 
                to_merge = k.to_frame().transpose()
                to_merge['CURRENT_GAME_ID'] = to_merge['CURRENT_GAME_ID'].apply(int)
                tuples_a['CURRENT_GAME_ID'] = tuples_a['CURRENT_GAME_ID'].apply(int)
                tuples_a = pd.merge(tuples_a, to_merge, on=['CURRENT_GAME_ID'], sort=False)
            
            tuples_b = pd.DataFrame()
            first_tuple_b = previous_games_b.iloc[0]
            first_tuple_b = first_tuple_b.add_suffix('_PREV_GAME_B_' + str(0))
            first_tuple_b['CURRENT_GAME_ID'] = i['GAME_ID_HOME'] 
            tuples_b = tuples_b.append(first_tuple_b)
            previous_games_b = previous_games_b.iloc[1:]

            for index5, l in previous_games_b.iterrows(): 
                l = l.add_suffix('_PREV_GAME_B_' + str(index5))
                l['CURRENT_GAME_ID'] = i['GAME_ID_HOME']
                to_merge = l.to_frame().transpose()
                to_merge['CURRENT_GAME_ID'] = to_merge['CURRENT_GAME_ID'].apply(int)
                tuples_b['CURRENT_GAME_ID'] = tuples_b['CURRENT_GAME_ID'].apply(int) 
                tuples_b = pd.merge(tuples_b, to_merge, on=['CURRENT_GAME_ID'], sort=False)

            tuples = tuples.append(pd.merge(tuples_a, tuples_b, on=['CURRENT_GAME_ID'], sort=False))
        
        print ('COMPLETED ITERATION: ', iteration)
        iteration += 1
    
    return tuples

def GetWinPercentage(n, df):
    print("***** Calculating Win Percentage *****")
    iteration = 0
    tuples = []

    for index_i, row_i in df.iterrows():
        tuple_ = pd.DataFrame({'CURRENT_GAME_ID': [0], 'WIN_PERCENTAGE_A': [0.0], 'WIN_PERCENTAGE_B': [0.0]})
        team_a = row_i['TEAM_NAME_HOME']
        team_b = row_i['TEAM_NAME_AWAY']
        game_date = row_i['GAME_DATE_EST_HOME']
        team_a_wins = 0.0
        team_a_losses = 0.0
        team_b_wins = 0.0
        team_b_losses = 0.0

        # Get win percentage for team a
        count_a = 0
        for index_j, row_j in df.iloc[index_i:].iterrows():
            if team_a == row_j['TEAM_NAME_HOME'] and row_j['GAME_DATE_EST_HOME'] < game_date:
                if row_j['WIN_HOME'] == 1:
                    team_a_wins += 1
                else:
                    team_a_losses += 1
                count_a += 1
            elif team_a == row_j['TEAM_NAME_AWAY'] and row_j['GAME_DATE_EST_HOME'] < game_date:
                if row_j['WIN_HOME'] == 0:
                    team_a_wins += 1
                else:
                    team_a_losses += 1
                count_a += 1
            if count_a >= n:
                break
        
        if team_a_wins == 0 and team_a_losses == 0:
            win_perc_a = np.NaN
        else:
            win_perc_a = team_a_wins / (team_a_wins + team_a_losses)
        
        # Get win percentage for team b
        count_b = 0
        for index_j, row_j in df.iloc[index_i:].iterrows():
            if team_b == row_j['TEAM_NAME_HOME'] and row_j['GAME_DATE_EST_HOME'] < game_date:
                if row_j['WIN_HOME'] == 1:
                    team_b_wins += 1
                else:
                    team_b_losses += 1
                count_b += 1
            elif team_b == row_j['TEAM_NAME_AWAY'] and row_j['GAME_DATE_EST_HOME'] < game_date:
                if row_j['WIN_HOME'] == 0:
                    team_b_wins += 1
                else:
                    team_b_losses += 1
                count_b += 1
            if count_b >= n:
                break
        if team_b_wins == 0 and team_b_losses == 0:
            win_perc_b = np.NaN
        else:
            win_perc_b = team_b_wins / (team_b_wins + team_b_losses)
        tuple_.at[0, 'CURRENT_GAME_ID'] = row_i['GAME_ID_HOME']
        tuple_.at[0, 'WIN_PERCENTAGE_A'] = win_perc_a
        tuple_.at[0, 'WIN_PERCENTAGE_B'] = win_perc_b   
        tuples.append(tuple_)
        
        print ('COMPLETED ITERATION: ', iteration)
        iteration += 1
    
    return pd.concat(tuples)

def GetDefRtgHome(x):
    sum_rtg = 0
    suffixes = [c[8:] for c in x.index if 'OPPONENT' in c and '_A' in c]
    for suffix in suffixes:
        if x['OPPONENT' + suffix] == 'HOME':
            sum_rtg = sum_rtg + x['OFF_RTG_HOME' + suffix]
        elif x['OPPONENT' + suffix] == 'AWAY':
            sum_rtg = sum_rtg + x['OFF_RTG_AWAY' + suffix]

    return sum_rtg/LAST_N

def GetOffRtgHome(x):
    sum_rtg = 0
    suffixes = [c[8:] for c in x.index if 'OPPONENT' in c and '_A' in c]
    for suffix in suffixes:
        if x['OPPONENT' + suffix] == 'HOME':
            sum_rtg = sum_rtg + x['OFF_RTG_AWAY' + suffix]
        elif x['OPPONENT' + suffix] == 'AWAY':
            sum_rtg = sum_rtg + x['OFF_RTG_HOME' + suffix]

    return sum_rtg/LAST_N

def GetDefRtgAway(x): 
    sum_rtg = 0
    suffixes = [c[8:] for c in x.index if 'OPPONENT' in c and '_B' in c]
    for suffix in suffixes:
        if x['OPPONENT' + suffix] == 'HOME':
            sum_rtg = sum_rtg + x['OFF_RTG_HOME' + suffix]
        elif x['OPPONENT' + suffix] == 'AWAY':
            sum_rtg = sum_rtg + x['OFF_RTG_AWAY' + suffix]

    return sum_rtg/LAST_N

def GetOffRtgAway(x):
    sum_rtg = 0
    suffixes = [c[8:] for c in x.index if 'OPPONENT' in c and '_B' in c]
    for suffix in suffixes:
        if x['OPPONENT' + suffix] == 'HOME':
            sum_rtg = sum_rtg + x['OFF_RTG_AWAY' + suffix]
        elif x['OPPONENT' + suffix] == 'AWAY':
            sum_rtg = sum_rtg + x['OFF_RTG_HOME' + suffix]

    return sum_rtg/LAST_N

def GetNetRtgHome(x):
    return x['OFF_RTG_HOME'] - x['DEF_RTG_HOME']

def GetNetRtgAway(x):
    return x['OFF_RTG_AWAY'] - x['DEF_RTG_AWAY']

def GetNetRating(df):
    # Calculate OFF and DEF Rating 
    df['DEF_RTG_HOME'] = df.apply(GetDefRtgHome, axis=1)
    df['DEF_RTG_AWAY'] = df.apply(GetDefRtgAway, axis=1)
    df['OFF_RTG_HOME'] = df.apply(GetOffRtgHome, axis=1)
    df['OFF_RTG_AWAY'] = df.apply(GetOffRtgAway, axis=1)
    
    # Calculate NET 
    df['NET_RTG_HOME'] = df.apply(GetNetRtgHome, axis=1)
    df['NET_RTG_AWAY'] = df.apply(GetNetRtgAway, axis=1)

    return df

def ReduceDimensions(df):
    # Extract wanted columns
    cols = [c for c in df.columns if c == 'CURRENT_GAME_ID'
                                    or c == 'TEAM_NAME_HOME' 
                                    or c == 'TEAM_NAME_AWAY'
                                    or c == 'GAME_DATE_EST'
                                    or c == 'WIN_HOME'
                                    or c == 'NET_RTG_HOME'
                                    or c == 'NET_RTG_AWAY'
                                    or c == 'EXPERT_RANK_HOME'
                                    or c == 'EXPERT_RANK_AWAY'
            ]
    return df[cols]

def EncodeTeamName(df):
    print (df)
    # Get TEAM_NAME columns
    cols = [c for c in df.columns if c[:9] == 'TEAM_NAME']
    # Encode team names
    for col in cols:        
        temp = pd.get_dummies(df[col])
        #df = df.drop(columns=[col])
        df = pd.concat([df, temp], axis=1)
    return df


# Main
df_2019 = ImportData('../data/raw_stats/2018-2019_team_stats.csv', '../data/raw_stats/2018-2019_summary.csv')
df_2018 = ImportData('../data/raw_stats/2017-2018_team_stats.csv', '../data/raw_stats/2017-2018_summary.csv')

df = pd.DataFrame()
df = df_2019
df = df.append(df_2018, ignore_index=True)
df = df.head(300)

LAST_N = 2
df= LastNCommonOpponents(2, df)
df = df.drop_duplicates()
print df
df = GetNetRating(df)
df = ReduceDimensions(df)
print df



