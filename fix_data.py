import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/malicious_phish.csv")

def fix_url(row):
    url = row['url']
    if row['type'] == 'benign':
        if not url.startswith('http'):
            # randomly assign http or https and www
            url = f"https://www.{url}" if np.random.rand() > 0.5 else f"http://{url}"
            # give it a standard path so it matches malicious
            if '/' not in url:
                url = f"{url}/" if np.random.rand() > 0.5 else url
    return url

np.random.seed(42)
df['url'] = df.apply(fix_url, axis=1)

print('Benign starts with http:', df[df['type']=='benign']['url'].str.startswith('http').mean())
print('Malicious starts with http:', df[df['type']!='benign']['url'].str.startswith('http').mean())

df.to_csv("data/raw/malicious_phish_fixed.csv", index=False)
