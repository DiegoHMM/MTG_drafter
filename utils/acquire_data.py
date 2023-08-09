import pandas as pd
# Local
from utils.scryfall_functions import mount_cards_df
from utils.handle_data import *


if __name__ == '__main__':
    #Build df all cards from set
    SET = 'ltr'
    URL = "https://api.scryfall.com/cards/search?q=set="
    if mount_cards_df(URL, SET):
        print("Handled_data saved as CSV")
    else:
        print("Something went wrong")

    ##Read df all cards from set
    card_attr_df = read_data('Handled_data/'+SET+'_cards_data.csv')

    ##filter best players
    output_file = "Handled_data/draft_data_best_players.csv"
    file_path = "raw_data/draft_data_public.LTR.PremierDraft.csv"
    ###get types from csv
    conversion_dict = get_integer_columns_and_dtype(file_path)  # Assuming get_integer_columns_and_dtype function defined earlier
    filter_best_players(file_path, output_file, conversion_dict)

    ##process deck data
    input_file = "Handled_data/draft_data_best_players.csv"
    output_file_stops = "Handled_data/df_stops.csv"
    conversion_dict = get_integer_columns_and_dtype(input_file)
    process_deck_data(input_file, output_file_stops, conversion_dict)

    ## add stops to dataframe
    best_players_file = "../Handled_data/draft_data_best_players.csv"
    stops_file = "../Handled_data/df_stops.csv"
    output_file = "../Handled_data/draft_data_best_players_sc.csv"
    conversion_dict = get_integer_columns_and_dtype(
        best_players_file)  # Assuming get_integer_columns_and_dtype function defined earlier
    add_stops_to_dataframe(best_players_file, stops_file, output_file, conversion_dict)