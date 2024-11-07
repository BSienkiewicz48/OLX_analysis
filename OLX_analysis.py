import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from collections import Counter
import openai


api_key = st.secrets["api_key"]


# Ustawienia strony
st.set_page_config(
    page_title="OLX Najkorzystniejsze Oferty",
    page_icon="💰",
    initial_sidebar_state="expanded",
)

st.title("💰 Wyszukiwarka Najkorzystniejszych Ofert na OLX")
st.markdown("""
Aplikacja **OLX Najkorzystniejsze Oferty** umożliwia analizę i porównanie ofert dostępnych na portalu OLX pod kątem stosunku jakości do ceny. Dzięki naszej aplikacji możesz szybko znaleźć najlepsze oferty w interesujących Cię segmentach rynku.
""")

# Pobierz URL od użytkownika
user_input = st.text_input("Wprowadź link do strony OLX:")

if not user_input:
    st.warning("Proszę wprowadzić link do wyniku wyszukiwania olx, upewnij się że przefiltrowałeś odpowiednio ogłoszenia - zgodnie z twoimi oczekiwaniami")
    st.stop()

# Sprawdź, czy URL jest z domeny OLX
if user_input:
    parsed_url = urlparse(user_input)
    if parsed_url.netloc.endswith("olx.pl"):
        # Zastąp base_url wartością z input boxa
        base_url = user_input
        st.write("Podany URL jest prawidłowy dla domeny OLX.")

        # Rozdziel URL na części
        url_parts = list(urlparse(base_url))
        query = parse_qs(url_parts[4])
    else:
        st.error("Podany URL nie należy do domeny OLX. Wprowadź poprawny link.")


if not st.button("Kliknij, aby kontynuować"):
    st.warning("Po wprowadzeniu linku kliknij przycisk powyżej, aby kontynuować.")
    st.stop()

#Poniżej fragment przygotowujący dane:

# URL strony, którą chcesz zczytać
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

# Przypisuję klaster na podstawie kolumny 'Cena'
prices = filtered_data[['Cena']].copy()
kmeans = KMeans(n_clusters=3, random_state=0)
filtered_data['Segment'] = kmeans.fit_predict(prices)

# Nazwanie segmentów na podstawie centroidów
centroids = kmeans.cluster_centers_
segment_labels = ['niski', 'średni', 'wysoki']
sorted_segments = sorted(range(3), key=lambda x: centroids[x][0])

# Mapowanie klastrów do nazw segmentów
filtered_data['Segment'] = filtered_data['Segment'].map({sorted_segments[i]: segment_labels[i] for i in range(3)})

# Obliczanie najważniejszych wartości dla każdego segmentu
summary_stats = filtered_data.groupby('Segment')['Cena'].agg(['mean', 'median', 'min', 'max']).reset_index()

#
#Wyświetlanie analizy
#

st.header("📊 Statystyki walidacji")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Liczba ogłoszeń", total_observations)
col2.metric("Usunięte outliery", number_of_outliers)
col3.metric("Usunięte wiersze <10% mediany", number_below_threshold)
col4.metric("Łącznie usuniętych", total_removed)


# Nagłówek sekcji wykresu
st.header("📈 Rozkład Cen w Segmentach")

# Tworzenie wykresu
plt.figure(figsize=(12, 10))
sns.violinplot(data=filtered_data, x='Segment', y='Cena', palette="muted", inner="box")

# Tytuł i etykiety wykresu
plt.title('Im szerszy wykres tym więcej ogłoszeń znajduje się w danej cenie)', fontsize=16)
plt.xlabel('Segment', fontsize=14)
plt.ylabel('Cena (PLN)', fontsize=14)

# Wyświetlenie wykresu
st.pyplot(plt)

# Przygotowanie danych do tabeli i zaokrąglenie do 2 miejsc po przecinku
styled_summary = summary_stats.rename(columns={
    'Segment': 'Segment',
    'mean': 'Średnia (PLN)',
    'median': 'Mediana (PLN)',
    'min': 'Min (PLN)',
    'max': 'Max (PLN)'
}).copy()
styled_summary[['Średnia (PLN)', 'Mediana (PLN)', 'Min (PLN)', 'Max (PLN)']] = styled_summary[['Średnia (PLN)', 'Mediana (PLN)', 'Min (PLN)', 'Max (PLN)']].applymap(lambda x: f"{x:.2f}")

# Stylizacja tabeli za pomocą HTML i CSS
st.write("### Statystyki dla poszczególnych segmentów")
table_html = styled_summary.to_html(index=False, classes="styled-table")

st.markdown(
    """
    <style>
    .styled-table {
        font-family: Arial, sans-serif;
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 16px;
        min-width: 400px;
        width: 100%;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        text-align: center;
    }
    .styled-table thead tr {
        background-color: #009879;
        color: #ffffff;
        text-align: center;
    }
    .styled-table th, .styled-table td {
        padding: 12px 15px;
        text-align: center;
    }
    .styled-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    .styled-table tbody tr:last-of-type {
        border-bottom: 2px solid #009879;
    }
    .styled-table tbody tr:hover {
        background-color: #d1e7dd;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Wyświetlenie tabeli w Streamlit
st.markdown(table_html, unsafe_allow_html=True)




client = openai.OpenAI(api_key=api_key)

# Wyszukiwany elemnt
search_terms_string = ' '.join(search_terms)

# Filtrować dane, aby uzyskać tylko te, w których kolumna 'Segment' ma wartość 'niski'
filtered_data_niski = filtered_data[filtered_data['Segment'] == 'niski'].copy()

# Dodanie kolumny z grupami opartymi na 20- centylach
filtered_data_niski['Percentyl'] = pd.qcut(filtered_data_niski['Cena'], q=5, labels=False)

# Funkcja do zliczania słów w grupie
def count_words_in_group(group):
    word_counter = Counter()
    for text in group['Tekst']:
        words = re.findall(r'\b\w+\b', text.lower())
        words = [word for word in words if len(word) > 1]  # Filtruj słowa krótsze niż 2 litery
        word_counter.update(words)
    return word_counter

# Zliczanie słów dla każdej grupy i zapisywanie wyników do DataFrame
group_word_counts = {}
for group_number in range(5):
    group = filtered_data_niski[filtered_data_niski['Percentyl'] == group_number]
    word_counter = count_words_in_group(group)
    common_words = word_counter.most_common(3)  # Najczęściej występujące słowa
    group_word_counts[group_number] = common_words

# Tworzenie DataFrame z wynikami zliczania słów
group_word_counts_df = pd.DataFrame.from_dict(group_word_counts, orient='index')
group_word_counts_df.columns = ['Word1', 'Word2', 'Word3']

# Usuń liczby wystąpień, pozostawiając same słowa
group_word_counts_df['Word1'] = group_word_counts_df['Word1'].apply(lambda x: x[0])
group_word_counts_df['Word2'] = group_word_counts_df['Word2'].apply(lambda x: x[0])
group_word_counts_df['Word3'] = group_word_counts_df['Word3'].apply(lambda x: x[0])

# W dalszej części kodu: użycie klasyfikacji
response_word = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Ma to być prosta klasyfikacja, tylko jedno słowo niepasujące do reszty."},
        {"role": "user", "content": (
            f"Sprawdź które słowo z tabeli {group_word_counts_df} nie pasuje do wyszukiwanej frazy: "
            f"{search_terms_string}. Napisz tylko jedno słowo z tabeli - to które nie pasuje. "
            f"Słowa jak gwarancja, czy stan lub model, wersja, mogą wystąpić w wyszukiwanej frazie. "
            f"Słowo które może zawierać się w zestawie do {search_terms_string} to go zostawiaj."
        )}
    ]
)


odpowiedzi_word = response_word.choices[0].message.content.strip()

print(odpowiedzi_word)

# Usuń wiersze, które zawierają słowo 'odpowiedzi_word' w kolumnie 'Tekst'
filteredV2_data_niski = filtered_data_niski[~filtered_data_niski['Tekst'].str.contains(odpowiedzi_word, case=False, na=False)]

import requests
from bs4 import BeautifulSoup

# Funkcja do scrapowania danych z podanego linku
def scrape_data(link):
    response = requests.get(link)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        elements = soup.find_all(class_='css-1o924a9')
        return [element.get_text(strip=True) for element in elements]
    else:
        return []

# Przechodzenie przez każdą wartość w kolumnie 'Link' i scrapowanie danych
filteredV2_data_niski.loc[:, 'tresc_ogloszenia'] = filteredV2_data_niski['Link'].apply(scrape_data)
filteredV2_data_niski = filteredV2_data_niski[filteredV2_data_niski['tresc_ogloszenia'].map(lambda d: len(d) > 0)]
data_to_show_st = filteredV2_data_niski.drop(columns=['Segment', 'Percentyl'])

if st.button("Pokaż tabelę"):
    st.dataframe(data_to_show_st, use_container_width=True)
st.markdown("Usunięto ogłoszenia z "+odpowiedzi_word)