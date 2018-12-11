from nba_py import game
from nba_py import player
from nba_py import team
import nba_py
import pandas as pd
import pprint as pp
import csv
import time

box_scores_player_stats = []
box_scores_team_stats = []
box_scores_summaries = []

def GetAllBoxScores(next_n_days, start_month, start_day, start_year):
    print("GETTING ALL BOXSCORES")
    

    for i in range(next_n_days, 0, -1):
        # Request Scoreboard
        scoreboard = nba_py.Scoreboard(month=start_month, day=start_day, year=start_year, league_id='00', offset=i)
        box_scores_summaries.append(scoreboard.game_header())
        time.sleep(1)
        
        print("SCOREBOARD REQUEST COMPLETE: " + str(i))

        for game_id in scoreboard.available()['GAME_ID']:
            # Request Boxscore
            player_stats = game.Boxscore(game_id).player_stats()
            box_scores_player_stats.append(player_stats)
            time.sleep(1)

            team_stats = game.Boxscore(game_id).team_stats()
            box_scores_team_stats.append(team_stats)
            time.sleep(1)
            
            print("BOXSCORE REQUEST COMPLETE")



GetAllBoxScores(49, 10, 16, 2018)

df = pd.concat(box_scores_player_stats)            
df.to_csv('2018-2019_player_stats2.csv')

df = pd.concat(box_scores_team_stats)            
df.to_csv('2018-2019_team_stats2.csv')

df = pd.concat(box_scores_summaries)            
df.to_csv('2018-2019_summary2.csv')
