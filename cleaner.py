import pandas as pd
import re


def cleaning(text):
    text = str(text).lower()
    text = " ".join([word for word in text.split() if '#' not in word and '@' not in word])
    text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    text = re.sub(r'''      
                   \W+       # Bir veya daha fazla sözcük olmayan karakter
                   \s*       # artı sıfır veya daha fazla boşluk karakteri,
                   ''',
                  ' ',
                  text,
                  flags=re.VERBOSE)
    text = re.sub("\d+", "", text)
    return text


def turkish_char(text):
    translation_table = str.maketrans("ğĞıİöÖüÜşŞçÇ", "gGiIoOuUsScC")
    result = text.translate(translation_table)
    return result


def data_cleaning():
    try:
        df = pd.read_csv('data/teknofest_train_final.csv', sep='|')
        text = df.text.apply(turkish_char)
        df['text'] = text.apply(cleaning)
        df.to_csv('data/clean_data.csv', index=False)
    except Exception as e:
        print('Alınan data temizlenirken bir hata oluştu', e)
