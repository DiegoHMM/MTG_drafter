Problem Statement
========================================

 Given a list of cards, the task is to select one card. With each new list, select a new card taking into account the cards chosen previously. This is a typical use-case for card games like Magic: The Gathering, where players need to build a deck by choosing one card at a time from a given list.
# Dataset
 The dataset consists of card data and deck usage statistics:

Card data: Obtained from the Scryfall API, it includes detailed information about each card. [Scryfall](https://scryfall.com/docs/api).

Deck usage statistics: Provided by the 17 lands API, it contains statistics about how frequently certain cards are used in decks by top-ranked players.  [17 lands](https://www.17lands.com/public_datasets)

# Pre-requisites:
Before you begin, ensure you have met the following requirements:

1. Python installed on your local machine.
2. Required Python libraries installed. You can install them using pip install -r requirements.txt.

# Execution Steps:

## Step 0: Get raw data:
   Download from [17 lands](https://www.17lands.com/public_datasets) the file: `draft_data_public...PremierDraft.csv` and put it in the `raw_data` folder 

## Step 1: Generate Dataset
```bash
execute: python build_dataset.py
```
This script will generate the following files:
    
1. `Data/ltr_cards_data.csv:` This file contains the data for all the cards in the collection.'

2.  `Data/draft_data_best_players.csv:` This file contains the data of decks which are in the top rankings: platinum, diamond, and mythic.
    
3. `Data/df_stops.csv:` This dataframe contains only the last cards chosen, already added to the pool.
    
4.  `Data/draft_data_best_players_stop.csv:` This dataframe contains only the cards that are among the rankings: {platinum, diamond, mythic} and with the last cards chosen already added to the pool.