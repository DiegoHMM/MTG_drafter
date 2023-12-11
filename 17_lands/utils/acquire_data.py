import pandas as pd
import sys
import os

# Local
from scryfall_functions import mount_cards_df
from handle_data import *

if __name__ == '__main__':
    # Verifica se os argumentos de linha de comando foram fornecidos
    if len(sys.argv) < 3:
        print("Usage: python acquire_data.py <set> <filename>")
        sys.exit(1)

    SET = sys.argv[1]
    filename = sys.argv[2]
    URL = "https://api.scryfall.com/cards/search?q=set="

    directory ="../handled_data"

    # Monta DataFrame de cartas
    if mount_cards_df(directory,URL, SET):
        print("handled_data saved as CSV")
    else:
        print("Something went wrong")

    # Leitura do DataFrame de todas as cartas do conjunto
    card_attr_df = read_data(directory + f'/{SET}_cards_data.csv')

    # Filtra os melhores jogadores
    print("Filtrando melhores jogadores...")
    output_file = directory + f"/draft_data_best_players_{SET}.csv"
    file_path = f"../raw_data/{filename}"
    conversion_dict = get_integer_columns_and_dtype(file_path)
    filter_best_players(file_path, output_file, conversion_dict)

    # Processa os dados do deck
    print("Processando melhores decks...")
    input_file = directory + f"/draft_data_best_players_{SET}.csv"
    output_file_stops = directory + f"/df_stops_{SET}.csv"
    conversion_dict = get_integer_columns_and_dtype(input_file)
    process_deck_data(input_file, output_file_stops, conversion_dict)

    # Adiciona paradas ao DataFrame
    print("Adicionando ultimo pick a pool final...")
    best_players_file = directory + f"/draft_data_best_players_{SET}.csv"
    stops_file = directory + f"/df_stops_{SET}.csv"
    output_file = directory + f"/draft_data_best_players_sc_{SET}.csv"
    conversion_dict = get_integer_columns_and_dtype(best_players_file)
    add_stops_to_dataframe(best_players_file, stops_file, output_file, conversion_dict)
