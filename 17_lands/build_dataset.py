from utils.handle_data import get_integer_columns_and_dtype, get_cards_metadata, process_and_save_in_batches
import pandas as pd
import numpy as np
import random
import sys

def extract_card_names(row, prefix):
    """ Extrai os nomes das cartas com base no prefixo e retorna uma lista de nomes. """
    return [name[len(prefix):] for name, value in row.items() if name.startswith(prefix) and value]

if __name__ == '__main__':
    # Seed for reproducibility
    np.random.seed(42)

    if len(sys.argv) < 2:
        print("Usage: python build_dataset.py <set>")
        sys.exit(1)

    SET = sys.argv[1]

    print("Lendo dados brutos...")
    d_type = get_integer_columns_and_dtype('handled_data/draft_data_best_players_'+SET+'.csv')
    raw_data = pd.read_csv('handled_data/draft_data_best_players_'+SET+'.csv', dtype=d_type)

    #groupby draft_id
    draft_id_group = raw_data.groupby('draft_id')

    #remove all group with less than 42 rows
    print("Removendo drafts incompletos...")
    draft_id_group = draft_id_group.filter(lambda x: len(x) == 42)


    # Criação das novas colunas 'pool' e 'pack'
    print("Criando colunas 'pool' e 'pack'...")
    draft_id_group['pool'] = draft_id_group.apply(lambda row: extract_card_names(row, 'pool_'), axis=1)
    draft_id_group['pack'] = draft_id_group.apply(lambda row: extract_card_names(row, 'pack_card_'), axis=1)

    # Resume os dados
    filtered_df = draft_id_group[['draft_id', 'rank','pick_number', 'pack_number', 'pick','pack', 'pool']]


   

    # Get a list of unique group identifiers (i.e., unique 'draft_id's)
    unique_ids = filtered_df['draft_id'].unique()
    # Sample 20% of the unique identifiers for the test set
    test_ids = np.random.choice(unique_ids, size=int(len(unique_ids) * 0.20), replace=False)
    # Create a new list for the remaining ids (those not in test set)
    remaining_ids = [id for id in unique_ids if id not in test_ids]
    # Sample 25% of the remaining identifiers for the validation set (0.25 * 0.8 = 0.2)
    val_ids = np.random.choice(remaining_ids, size=int(len(remaining_ids) * 0.25), replace=False)
    # The rest (60% of total) goes to the train set
    train_ids = [id for id in remaining_ids if id not in val_ids]


    # Create the sets based on the sampled identifiers
    print("Salvando dados...")
    test_set = filtered_df[filtered_df['draft_id'].isin(test_ids)]
    val_set = filtered_df[filtered_df['draft_id'].isin(val_ids)]
    train_set = filtered_df[filtered_df['draft_id'].isin(train_ids)]


    #save data
    test_set.to_csv('dataset/test_set.csv', index=False)
    val_set.to_csv('dataset/val_set.csv', index=False)
    train_set.to_csv('dataset/train_set.csv', index=False)

    print("Done!")