import pandas as pd
import requests
import os
import numpy as np

def mount_cards_df(URL, SET):
    card_attr_list = []
    while URL:
        r = requests.get(url=URL + SET)
        r.raise_for_status()
        data = r.json()

        # Process each card
        for card in data['data']:
            card_attr = {
                'name': card['name'],
                'rarity': card['rarity'],
                'color_identity': 'M' if len(card['color_identity']) > 1 else (card['color_identity'][0] if card['color_identity'] else 'T'),
                'cmc': card['cmc'],
                'colors': 'M' if len(card['colors']) > 1 else (card['colors'][0] if card['colors'] else 'T'),
            }
            card_attr_list.append(card_attr)

        # Check if there are more cards
        URL = data['next_page'] if data['has_more'] else None

    # Save the data
    os.makedirs('Data', exist_ok=True)
    df = pd.DataFrame(card_attr_list)
    df.to_csv(f'Data/{SET}_cards_data.csv', index=False)

    return True



