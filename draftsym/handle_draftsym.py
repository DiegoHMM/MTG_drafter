import pandas as pd
import pickle


def format_pkl_to_csv(data_pkl):
    # Transformar em DataFrame
    data = []
    for draft_index, draft in enumerate(data_pkl):
        pool = []
        for round_index, round_ in enumerate(draft):
            pick = round_[0]
            available = round_[1:]
            data.append([draft_index+1, round_index+1, pick, available, list(pool)])
            pool.append(pick)
        # Adicionando linha ao final de cada draft apenas para atualizar a pool
        data.append([draft_index+1, round_index+2, None, [], list(pool)])

    df = pd.DataFrame(data, columns=['#_draft', '#_round', 'pick', 'available', 'pool'])

    return df



if __name__ == '__main__':
    with open('/home/diego/Documentos/MTG_drafter/draftsym/dataset/drafts_train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('/home/diego/Documentos/MTG_drafter/draftsym/dataset/drafts_test.pkl', 'rb') as f:
        test_data = pickle.load(f)


    df_train = format_pkl_to_csv(train_data)
    df_test = format_pkl_to_csv(test_data)

    df_train.to_csv('/home/diego/Documentos/MTG_drafter/dataset/draftsym/train.csv')
    df_test.to_csv('/home/diego/Documentos/MTG_drafter/dataset/draftsym/teste.csv')