import pandas as pd
import numpy as np

def read_data(path):
    data = pd.read_csv(path)
    return data

def get_integer_columns_and_dtype(file_path):
    # Read first line of csv file and get integer columns
    df = pd.read_csv(file_path, nrows=1)

    # Get columns that start with 'pool_' or 'pack_card_'
    columns = [col for col in df.columns if 'pool_' in col or 'pack_card_' in col]

    # Create a dictionary that maps each integer column to np.int8,
    # 'user_game_win_rate_bucket' to np.float32, and 'pick' to 'category'
    conversion_dict = {item: np.int8 for item in columns}
    conversion_dict['user_game_win_rate_bucket'] = np.float32
    conversion_dict['pick'] = 'category'

    return conversion_dict
def filter_best_players(file_path, output_file, conversion_dict, chunksize=100000):
    best_players_pick_list = []
    df = pd.read_csv(file_path, chunksize=chunksize, dtype=conversion_dict)

    for chunk in df:
        best_players_pick_list.append(
            chunk[(chunk['rank'] == 'platinum') | (chunk['rank'] == 'diamond') | (chunk['rank'] == 'mythic')])

    # Concat and create new .csv file
    best_players_df = pd.concat(best_players_pick_list)
    best_players_df.to_csv(output_file, index=False)
    print("Best players data saved as CSV")


def process_deck_data(input_file, output_file_stops, conversion_dict):
    best_players_df = pd.read_csv(input_file, dtype=conversion_dict)

    # Get the last pick of last pack
    df_deck = best_players_df[(best_players_df['pack_number'] == 2) & (best_players_df['pick_number'] == 13)]
    print("Number of decks: ", len(df_deck))
    df_stops = df_deck.copy(deep=True)

    # Zero the current pack cards (no pack has been opened)
    df_stops.loc[:, df_stops.columns.str.startswith('pack_card_')] = 0

    # Add the last opened card to the deck pool
    picked_cards = df_stops['pick'].unique()
    # Add last pick (#15) as keyword: "last"
    df_stops['pick_number'] = 14
    df_stops['pick'] = 'last'
    for card in picked_cards:
        df_stops.loc[df_stops['pick'] == card, 'pool_' + card] = df_stops[df_stops['pick'] == card]['pool_' + card] + 1

    df_stops.to_csv(output_file_stops, index=False)


def add_stops_to_dataframe(best_players_file, stops_file, output_file, conversion_dict):
    # Load dataframe
    best_players_pick_df = pd.read_csv(best_players_file, dtype=conversion_dict)
    # Load stops data
    df_stops = pd.read_csv(stops_file, dtype=conversion_dict)
    # Add Stops Values on big Dataframe
    best_players_pick_df = pd.concat([best_players_pick_df,df_stops])
    best_players_pick_df.to_csv(output_file, index=False)