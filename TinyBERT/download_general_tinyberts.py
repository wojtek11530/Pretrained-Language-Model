import os
import gdown
import zipfile

urls = {
    'General_TinyBERT_4': 'https://drive.google.com/uc?export=download&id=1dDigD7QBv1BmE6pWU71pFYPgovvEqOOj',
    'General_TinyBERT_6': 'https://drive.google.com/uc?export=download&id=1wXWR00EHK-Eb7pbyw0VP234i2JTnjJ-x'
}

output_path = os.path.join('data', 'models')
os.makedirs(output_path, exist_ok=True)

for name, url in urls.items():
    output_zip = os.path.join(output_path, f'{name}.zip')
    gdown.download(url, output_zip, quiet=False)
    with zipfile.ZipFile(output_zip, "r") as zip_ref:
        zip_ref.extractall(output_path)
    os.remove(output_zip)

print('Completed!')
