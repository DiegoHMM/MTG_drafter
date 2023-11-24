import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import ast

class MagicDataset(Dataset):
    def __init__(self, dataframe, modelo_w2v, card_dict):
        self.dataframe = dataframe
        self.modelo_w2v = modelo_w2v
        self.card_dict = card_dict
        
    def __len__(self):
        return len(self.dataframe)
    
    @staticmethod
    def nome_para_embedding(nome, modelo_w2v):
        return modelo_w2v.wv[nome]
    
    import numpy as np

    @staticmethod
    def embedding_para_nome(embedding, modelo_w2v):
        # Se o embedding for composto inteiramente por 0s, retorna 0
        if np.all(embedding == 0):
            return 0

        # Encontra as palavras mais similares. Retorna uma lista onde o primeiro item é o mais similar
        similar = modelo_w2v.wv.similar_by_vector(embedding, topn=1)
        return similar[0][0]  # Retorne apenas o nome da carta mais similar


    
    @staticmethod
    def string_para_lista(s):
        try:
            return ast.literal_eval(s)
        except (SyntaxError, ValueError):
            return []
        
    @staticmethod
    def criar_mascara_disponibilidade(available, cards_dict):
        mascara = torch.zeros(265)  # Supondo que há 265 cartas possíveis
        card_to_index = {card: idx for idx, card in enumerate(cards_dict['cards_list'])}
        
        for carta in available:
            index = card_to_index.get(carta, None)
            if index is not None:
                mascara[index] = 1
            
        return mascara



    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        pool = MagicDataset.string_para_lista(row['pool'])
        available = MagicDataset.string_para_lista(row['available'])
        mascara = MagicDataset.criar_mascara_disponibilidade(available, self.card_dict)
        
        pool_embedding = [MagicDataset.nome_para_embedding(carta, self.modelo_w2v) for carta in pool]
        while len(pool_embedding) < 45:
            pool_embedding.append([0]*50)

        pool_embedding = np.array(pool_embedding)
        X = torch.tensor(pool_embedding, dtype=torch.float32)
        
        # Agora, y será o índice da carta e não o embedding
        y = row['label']
        y = torch.tensor(y, dtype=torch.long)
        
        return X, y, mascara

