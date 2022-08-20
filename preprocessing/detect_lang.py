import numpy as np
import pandas as pd
import dask.dataframe as dd
import fasttext

# from pandarallel import pandarallel
# pandarallel.initialize(progress_bar=False)


def detect_non_english(text):
    lang, score = langid.classify(text)
    if lang != "en":
        return 1
    else:
        return 0

def detect_non_english_dask(data):
    return data.apply(detect_non_english)

def remove_newlines(txt):
    return txt.replace("\n", " ")

if __name__ == "__main__":
    df = pd.read_parquet("processed_data/joined.parquet.snappy")
    ddf = dd.from_pandas(df, npartitions=4)
    txts = ddf.r_text.map(remove_newlines, meta=(None, int)).compute()

    pretrained_model_path = "./ftt_model/lid.176.bin"
    fmodel = fasttext.load_model(pretrained_model_path)
    res = np.asarray(fmodel.predict(txts.tolist())[0]).reshape(-1)
    res = np.where(res != '__label__en', 1, 0)
    df['is_english'] = res
    
    df.to_parquet("processed_data/joined_data_lang_detected.parquet", index=False, compression="snappy")
    
    
    
