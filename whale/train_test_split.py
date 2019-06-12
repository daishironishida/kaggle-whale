import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from .dataset import DATA_ROOT


def make_folds(seed: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_ROOT / 'train.csv', index_col="Image")

    df_identified = df[df.Id != "new_whale"]
    df_train, df_val = train_test_split(df_identified, test_size=0.05, random_state=seed)
    num_items = df_train.Id.value_counts()
    not_paired = []
    for idx, row in df_train.iterrows():
        if num_items[row.Id] == 1:
            not_paired.append(idx)

    df_val = df_val.append(df_train.loc[not_paired])
    df_train = df_train.drop(not_paired)

    return df_identified, df_train, df_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    df_identified, df_train, df_val = make_folds(seed=args.seed)
    df_identified.to_csv('df_identified.csv')
    df_train.to_csv('df_train.csv')
    df_val.to_csv('df_val.csv')

if __name__ == '__main__':
    main()
    
    
    
    
    
    
