import itertools
import numpy as np
import pandas as pd
import random

from keras.models import Sequential
from keras.layers import Dense

# Define constants
ROSTER = ['QB','RB1','RB2','WR1','WR2','WR3','TE','FLEX','DST']
DATA_POINTS = ['DraftKingsSalary','UpcomingOpponentRank','UpcomingOpponentPositionRank']
DF_COLUMNS = [x + '_' + y for x in ROSTER for y in DATA_POINTS] + ['TotalPoints']

def main():
  
  all_cols = ['Week', 'Name','DraftKingsSalary','UpcomingOpponentRank','UpcomingOpponentPositionRank','FantasyPointsPerGameDraftKings']
  cols = ['Name','DraftKingsSalary','UpcomingOpponentRank','UpcomingOpponentPositionRank','FantasyPointsPerGameDraftKings']


  # TODO: load data in function, pass in year.
  salary_2020_qb = pd.read_csv('data/projections/2020/2020_qb_salary.csv')
  salary_2020_rb = pd.read_csv('data/projections/2020/2020_rb_salary.csv')
  salary_2020_wr = pd.read_csv('data/projections/2020/2020_wr_salary.csv')
  salary_2020_te = pd.read_csv('data/projections/2020/2020_te_salary.csv')
  salary_2020_dst = pd.read_csv('data/projections/2020/2020_dst_salary.csv')

  results_2020_qb = pd.read_csv('data/results/2020/2020_qb_results.csv')
  results_2020_rb = pd.read_csv('data/results/2020/2020_rb_results.csv')
  results_2020_wr = pd.read_csv('data/results/2020/2020_wr_results.csv')
  results_2020_te = pd.read_csv('data/results/2020/2020_te_results.csv')
  results_2020_dst = pd.read_csv('data/results/2020/2020_dst_results.csv')

  merged_2020_qb = salary_2020_qb.merge(results_2020_qb, how='inner', on=['Name', 'Week', 'Team', 'Position', 'Opponent'])
  merged_2020_rb = salary_2020_rb.merge(results_2020_rb, how='inner', on=['Name', 'Week', 'Team', 'Position', 'Opponent'])
  merged_2020_wr = salary_2020_wr.merge(results_2020_wr, how='inner', on=['Name', 'Week', 'Team', 'Position', 'Opponent'])
  merged_2020_te = salary_2020_te.merge(results_2020_te, how='inner', on=['Name', 'Week', 'Team', 'Position', 'Opponent'])
  merged_2020_dst = salary_2020_dst.merge(results_2020_dst, how='inner',on=['Name', 'Week', 'Team', 'Position', 'Opponent'])

  # Group each projection by week
  qb_2020_grouping = group_data_by_week(merged_2020_qb[all_cols], cols)
  rb_2020_grouping = group_data_by_week(merged_2020_rb[all_cols], cols)
  wr_2020_grouping = group_data_by_week(merged_2020_wr[all_cols], cols)
  te_2020_grouping = group_data_by_week(merged_2020_te[all_cols], cols)
  dst_2020_grouping = group_data_by_week(merged_2020_dst[all_cols], cols)

  train_data = []

  # Start by only using data from week 6 to week 16
  for i in range(6, 10):
    players = {}
    players['QB'] = qb_2020_grouping[i]
    players['RB1'] = rb_2020_grouping[i]
    players['RB2'] = rb_2020_grouping[i]
    players['WR1'] = wr_2020_grouping[i]
    players['WR2'] = wr_2020_grouping[i]
    players['WR3'] = wr_2020_grouping[i]
    players['TE'] = te_2020_grouping[i]
    players['FLEX'] = rb_2020_grouping[i] + wr_2020_grouping[i] + te_2020_grouping[i]
    players['DST'] = dst_2020_grouping[i]

    lineups = generate_random_lineups(players, 50000, 2000, 100)

    train_data.append(build_lineup_data_frame(lineups))

  build_model(pd.concat(train_data))


def build_model(train_data):
  """ Builds a model from the given training data.

    Args:
      train_data: list of data frames.
    Returns:
      Model: a model that can be used to predict lineups.
  """
  Y_train = train_data['TotalPoints']
  X_train = train_data.drop(columns=['TotalPoints'])

  model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
  ])

  model.compile(optimizer='adam', loss='mean_squared_error')



def build_lineup_data_frame(lineups):
  """ Builds a data frame from a list of lineups.

    Args:
      lineups: list of lineups.
    Returns:
      DataFrame: a data frame containing the lineups.
  """

  frames = []
  for lineup in lineups:
    row = {}
    total_points = 0
    for key, position in lineup.items():
      total_points += position['FantasyPointsPerGameDraftKings']
      for col in DATA_POINTS:
        row[key + '_' + col] = position[col]
      
    row['TotalPoints'] = total_points
    frames.append(pd.DataFrame([row], columns=DF_COLUMNS))
    
  df = pd.concat(frames)

  return df




def group_data_by_week(data, columns):
  """ Groups data by week.
    Args:
      data: list of data points.
    Returns
      dict: a dictionary where each numeric week maps to a list of data points.
  """

  grouping = {}
  for idx, row in data.iterrows():
    if row['Week'] not in grouping:
      grouping[row['Week']] = []

    # Don't use players that are not playing
    if isinstance(row['DraftKingsSalary'], float):
      grouping[row['Week']].append(row[columns].to_dict())

  return grouping

def generate_random_lineups(players, budget, max_budget_remaining, lineup_count=10000):
  """
  Generates all possible combinations of lineups that fit the budget and roster requirements.
  
  Args:
      players (dict): A dictionary where keys represent players, and values are lists of dictionaries containing player names and price.
      budget: The total budget available.
      max_budget_remaining: The maximum amount of the budget to be left unspent.
      lineup_count: The number of lineups to generate.
  
  Returns:
      list: A list of dictionaries representing all valid rosters.
  """

  # Create a list of ranges for each position
  player_indices_ranges = [range(len(p)) for p in players.values()]

  valid_lineups = []
  while len(valid_lineups) < lineup_count:
    player_indices = generate_random_list(player_indices_ranges)
    # Create a dictionary representing this combination

    # TODO: would love to make this generic
    lineup = {
       'QB': players['QB'][player_indices[0]],
       'RB1': players['RB1'][player_indices[1]],
       'RB2': players['RB2'][player_indices[2]],
       'WR1': players['WR1'][player_indices[3]],
       'WR2': players['WR2'][player_indices[4]],
       'WR3': players['WR3'][player_indices[5]],
       'TE': players['TE'][player_indices[6]],
       'FLEX': players['FLEX'][player_indices[7]],
       'DST': players['DST'][player_indices[8]]
    }

    lineup_inverse = {}
    for key, value in lineup.items():
       lineup_inverse.setdefault(value['Name'], []).append(key)

    # Check if any player is used more than once or if the lineup is already in the list of valid lineups
    if [key for key, values in lineup_inverse.items() if len(values) > 1] or lineup in valid_lineups:
      continue

    # Calculate the total cost for this lineup
    total_cost = sum(roster_spot['DraftKingsSalary'] for roster_spot in lineup.values())
      
    # Check if the total cost fits within the budget and the spending meets the minimum requirement
    if total_cost <= budget and total_cost >= (budget - max_budget_remaining):
      valid_lineups.append(lineup)

  return valid_lineups



def generate_random_list(ranges):
    """ Generates a set of random numbers that fit the range for each index in the given list of ranges.
    
    Args:
        ranges (list): A list of tuples representing the ranges for each index.
    Returns:
        set: A set of random numbers that fit the range for each index.
    """
    random_list = []
    
    for r in ranges:        
        # Generate a random number within the range and add it to the list.
        random_list.append(random.randint(r.start, len(r)-1))
    
    return random_list



if __name__ == "__main__":
    main()