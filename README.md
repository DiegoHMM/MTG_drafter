Problem Statement
========================================

 Given a list of cards, the task is to select one card. With each new list, select a new card taking into account the cards chosen previously. This is a typical use-case for card games like Magic: The Gathering, where players need to build a deck by choosing one card at a time from a given list.
# Dataset
 The dataset consists of card data and deck usage statistics:

Card data: Obtained from the Scryfall API, it includes detailed information about each card. [Scryfall](https://scryfall.com/docs/api).

Deck usage statistics: Provided by the 17 lands API, it contains statistics about how frequently certain cards are used in decks by top-ranked players.  [17 lands](https://www.17lands.com/public_datasets)

# Prerequisites:
Before you begin, ensure you have met the following requirements:

1. Python installed on your local machine.
2. Required Python libraries installed. You can install them using pip install -r requirements.txt.

# Execution Steps:

## Step 0: Get raw data:
   Download from [17 lands](https://www.17lands.com/public_datasets) the file: `draft_data_public...PremierDraft.csv` and put it in the `raw_data` folder 

## Step 1: Generate handled data:
```bash
On "utils" folder,

execute: python acquire_data.py
```
This script will generate the following files:
    
1. `Handled_data/ltr_cards_data.csv:` This file contains the data for all the cards in the collection.'

2.  `Handled_data/draft_data_best_players.csv:` This file contains the data of decks which are in the top rankings: platinum, diamond, and mythic.
    
3. `Handled_data/df_stops.csv:` This dataframe contains only the last cards chosen, already added to the pool.
    
4.  `Handled_data/draft_data_best_players_stop.csv:` This dataframe contains only the cards that are among the rankings: {platinum, diamond, mythic} and with the last cards chosen already added to the pool.

## Step 2: Generate dataset:

```bash
On "root",

execute: python build_dataset.py
```

### Data Structure:
The dataset captures multiple draft events, with each event being identified by a unique draft_id. Within each draft event, cards are selected in a sequence.

For each draft:

1. In round 1, cards `A, B, and C` are available. Card `A` is chosen.
2. In round 2, cards `D, E, and F` are available. Card `E` is selected.
3. In round 3, cards `G, H, and I` are available. Card `I` is chosen.
4. ...
### Representing this in our input and output data:
* Input X of the 3th round of the draft is a sequence of available cards in each of the previous rounds: `[[A,B,C],[D,E,F],[G,H,I]]`
* Output y is: `[A,E,I]`



