import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import re
import matplotlib.pyplot as plt
import numpy as np

#Poniżej fragment przygotowujący dane:

# URL strony, którą chcesz zczytać
base_url = "https://www.olx.pl/elektronika/gry-konsole/konsole/q-playstation-5/?search%5Bfilter_enum_state%5D%5B0%5D=used&search%5Bfilter_enum_version%5D%5B0%5D=playstation5"
sort_fragment = "created_at:desc"

# Rozdziel URL na części
url_parts = list(urlparse(base_url))
query = parse_qs(url_parts[4])

# Sprawdź, czy istnieje klucz 'search[order]' z wartością 'created_at:desc'
if "search[order]" not in query or query["search[order]"][0] != sort_fragment:
    query["search[order]"] = [sort_fragment]  # Dodaj fragment sortujący

# Ponownie koduj zapytanie z wszystkimi parametrami
url_parts[4] = urlencode(query, doseq=True)

# Zbuduj pełny URL
base_url = urlunparse(url_parts)

# Zczytanie zawartości pierwszej strony
response = requests.get(base_url)
soup = BeautifulSoup(response.content, 'html.parser')

# Znalezienie liczby stron z ogłoszeniami
total_pages_tag = soup.find_all('a', class_='css-1mi714g')
if total_pages_tag:
    total_pages = max([int(tag.get_text(strip=True)) for tag in total_pages_tag if tag.get_text(strip=True).isdigit()])
    if total_pages > 15:
        total_pages = 15
else:
    total_pages = 1

# Zmienna przechowująca zawartość wszystkich stron
combined_response = response.content

# Pętla iterująca przez kolejne strony
for page_number in range(2, total_pages + 1):
    # Ustaw lub zaktualizuj parametr `page`
    query["page"] = [str(page_number)]
    
    # Zakoduj ponownie zapytanie i zaktualizuj URL
    url_parts[4] = urlencode(query, doseq=True)
    page_url = urlunparse(url_parts)
    
    # Zczytaj zawartość bieżącej strony
    response = requests.get(page_url)
    if response.status_code == 200:
        combined_response += response.content  # Dodaj zawartość do `combined_response`
    else:
        print(f"Nie udało się pobrać strony {page_number}")

# Tworzenie obiektu BeautifulSoup z połączonej zawartości
soup = BeautifulSoup(combined_response, 'html.parser')

# Znalezienie wszystkich rodziców z określoną klasą wspólną dla linku i ceny
parents = soup.find_all('div', class_='css-u2ayx9')
# Przekształcenie listy linków i cen w DataFrame
link_data = []
for parent in parents:
    link = parent.find('a', class_='css-z3gu2d')
    price_tag = parent.find('p', class_='css-13afqrm')
    
    # Sprawdzenie, czy link i cena istnieją
    if link:
        href = link.get('href')
        text = link.get_text(strip=True)
        price = price_tag.get_text(strip=True) if price_tag else 'Brak ceny'

        link_data.append({'Link': f"https://www.olx.pl{href}", 'Tekst': text, 'Cena': price})

df = pd.DataFrame(link_data)

df['Cena'] = df['Cena'].str.replace('zł', '').str.replace(' ', '')
df['Do negocjacji'] = df['Cena'].str.contains('donegocjacji')
df['Cena'] = df['Cena'].str.replace('donegocjacji', '')
df['Cena'] = df['Cena'].str.replace(',', '.')
df = df[df['Cena'] != "Zamienię"]
df['Cena'] = pd.to_numeric(df['Cena'], errors='coerce') #wartości które nie mogą być float będą usunięte 
df = df.dropna(subset=['Cena']) # a wiersze dropnięte
df['Cena'] = df['Cena'].astype(float)
df['Do negocjacji'] = df['Do negocjacji'].astype(bool)
df = df.drop_duplicates(subset='Tekst', keep='first')

#Wyciąganie słów wyszukiwania
# Funkcja do wyciągania słów z wyszukiwanego tekstu
def extract_search_terms(url_parts):
    terms = []
    for part in url_parts:
        match = re.search(r'q-([a-zA-Z0-9\-]+)', part)
        if match:
            # Zamiana "-" na spacje, rozdzielanie i dodanie do listy
            terms.extend(match.group(1).replace('-', ' ').split())
    return terms
search_terms = extract_search_terms(url_parts)
df = df[df['Tekst'].apply(lambda x: all(word.lower() in x.lower() for word in search_terms))]

#Powyżej fragment przygotowujący dane
Links = df


#Poniżej czyszczenie danych:
# Usuń outliery z kolumny 'Cena'
q1 = Links['Cena'].quantile(0.25)
q3 = Links['Cena'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.8 * iqr
upper_bound = q3 + 1.5 * iqr

# Obliczanie liczby początkowych obserwacji
total_observations = len(Links)

# Zidentyfikuj outliery na podstawie przedziału (IQR)
outliers = Links[(Links['Cena'] < lower_bound) | (Links['Cena'] > upper_bound)]
number_of_outliers = len(outliers)

# Filtrowanie danych bez outlierów
filtered_data = Links[(Links['Cena'] >= lower_bound) & (Links['Cena'] <= upper_bound)]

# Oblicz medianę kolumny 'Cena' dla danych bez outlierów
median_price = filtered_data['Cena'].median()

# Ustaw granicę minimalną jako 10% mediany
min_threshold = 0.1 * median_price

# Zidentyfikuj wiersze poniżej 10% mediany
below_threshold = filtered_data[filtered_data['Cena'] < min_threshold]
number_below_threshold = len(below_threshold)

# Usuń wiersze, gdzie 'Cena' jest mniejsza niż 10% mediany
filtered_data = filtered_data[filtered_data['Cena'] >= min_threshold]

# Oblicz całkowitą liczbę usuniętych wierszy (outliery + poniżej 10% mediany)
total_removed = number_of_outliers + number_below_threshold

# Wyświetlanie liczby obserwacji i usuniętych wierszy
print(f'Liczba ogłoszeń branych pod uwagę przy analizie: {total_observations}')
print(f'Liczba usuniętych outlierów: {number_of_outliers}')
print(f'Liczba usuniętych wierszy poniżej 10% mediany: {number_below_threshold}')
print(f'Łączna liczba usuniętych wierszy: {total_removed}')


