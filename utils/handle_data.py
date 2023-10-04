import pandas as pd
import numpy as np
import gc
import pickle
import os

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


def get_cards_metadata(file_path):
    # Load the data
    meta_data_card = pd.read_csv(file_path)

    # Add an 'id' column
    meta_data_card['id'] = meta_data_card.index

    # One-hot encoding for 'rarity'
    meta_data_card = pd.concat([meta_data_card, pd.get_dummies(meta_data_card['rarity'], prefix='rarity').astype(int)], axis=1)
    meta_data_card.drop(['rarity'], axis=1, inplace=True)

    # One-hot encoding for 'colors'
    meta_data_card = pd.concat([meta_data_card, pd.get_dummies(meta_data_card['colors'], prefix='colors').astype(int)], axis=1)
    meta_data_card.drop(['colors'], axis=1, inplace=True)

    meta_data_card.drop(['color_identity'], axis=1, inplace=True)

    # Normalize 'cmc'
    meta_data_card['cmc'] = (meta_data_card['cmc'] - meta_data_card['cmc'].mean()) / meta_data_card['cmc'].std()

    return meta_data_card

# Função para obter características da carta
def get_card_features(card_name, df_card_features):
    card_row = df_card_features[df_card_features['name'] == card_name].squeeze()
    return card_row

def sequence_data_with_metadata(dataset, card_features_dict):
    """
    Prepare dataset with metadata and extract labels.

    Args:
    - dataset (pd.DataFrame): Dataset to be processed.
    - card_features_dict (dict): Dictionary mapping card names to their features.

    Returns:
    - X (list): List of input sequences.
    - y (list): List of output sequences (labels).
    """
    grouped = dataset.groupby('draft_id')

    X = []  # List to store input sequences
    y = []  # List to store output sequences (labels)

    for _, group in grouped:
        sequence = []
        for index, row in group.iterrows():
            # Identify which cards are available in this row
            available_cards = [col.split("pack_card_")[1] for col in row.filter(like='pack_card_').index if
                               row[col] == 1]
            # For each available card, get its features
            available_cards_features = [get_card_features(card_name, card_features_dict) for card_name in available_cards]
            sequence.append(available_cards_features)

            # Extract the 'pick' column as label
            y.append(row['pick'])

        X.append(sequence)

    return X, y


def process_and_save_in_batches(dataset_path, card_features_dict, base_filename, batch_size=21000):
    """Process the dataset in batches and save immediately to pickle files."""

    batch_num = 0  # To give unique names for each batch file

    # Use chunksize parameter to read in chunks
    for chunk in pd.read_csv(dataset_path, chunksize=batch_size):
        X, y = sequence_data_with_metadata(chunk, card_features_dict)

        # Save the processed chunk immediately
        with open(f"{base_filename}_X_{batch_num}.pkl", "wb") as f:
            pickle.dump(X, f)

        with open(f"{base_filename}_y_{batch_num}.pkl", "wb") as f:
            pickle.dump(y, f)

        # Delete the chunk's data from memory and force garbage collection
        del X, y
        gc.collect()

        batch_num += 1

    def load_from_batches(base_filename, prefix):
        """Loads data from multiple pickle files saved in batches.

        Args:
        - base_filename (str): The base name of the saved file.
        - prefix (str): The prefix used for saving files, e.g., 'X' or 'y'.

        Returns:
        - list: The combined data from all batches.
        """
        data = []
        batch_num = 0

        # Continue loading until a file is not found
        while os.path.exists(f"{base_filename}_{prefix}_{batch_num}.pkl"):
            with open(f"{base_filename}_{prefix}_{batch_num}.pkl", "rb") as f:
                batch_data = pickle.load(f)
                data.extend(batch_data)
            batch_num += 1

        return data