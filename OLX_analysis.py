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
from io import StringIO


api_key = st.secrets["api_key"]


# Ustawienia strony
st.set_page_config(
    page_title="OLX Analityka Ofert",
    page_icon="💰",
    initial_sidebar_state="expanded",
)

st.title("💰 Analiza rozkładu cen dla ofert na portalu ogłoszeniowym OLX")
st.markdown("""
Aplikacja **OLX** umożliwia analizę ofert dostępnych na portalu OLX pod kątem rozkładu cen. Dzięki aplikacji możesz szybko zorientować się w jakich cenach wystawiany jest dany przedmiot.
""")

# Domyślny link
default_link = "https://www.olx.pl/elektronika/gry-konsole/konsole/q-playstation-5/?search%5Bfilter_enum_state%5D%5B0%5D=used&search%5Bfilter_enum_version%5D%5B0%5D=playstation5"

# Zmień link za pomocą zmiennej user_input
user_input = input("Podaj nowy link (lub naciśnij Enter, aby użyć domyślnego): ").strip()
if user_input:
    link_to_use = user_input
else:
    link_to_use = default_link



if user_input:
    parsed_url = urlparse(user_input)
    if parsed_url.netloc.endswith("olx.pl"):
        # Zastąp base_url wartością z input boxa
        base_url = user_input
        st.session_state['base_url'] = base_url
        st.write("Podany URL jest prawidłowy dla domeny OLX.")
    else:
        st.error("Podany URL nie należy do domeny OLX. Wprowadź poprawny link.")
        st.stop()


# Przycisk do czyszczenia session_state
if st.button('Wygeneruj nowe dane'):
    st.session_state.clear()

# Rozdziel URL na części
url_parts = list(urlparse(base_url))
query = parse_qs(url_parts[4])

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

def scrapuj_dane():
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
        link = parent.find('a', class_='css-qo0cxu') #Jak nie działa to prawdopodobnie zmienili klasy na OLX @@@@@@
        price_tag = parent.find('p', class_='css-13afqrm') #Jak nie działa to prawdopodobnie zmienili klasy na OLX @@@@@@
        
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
    return df



# Sprawdzenie, czy dane są już w session_state
if 'Links' not in st.session_state:
    st.session_state.Links = scrapuj_dane()

Links = st.session_state.Links

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






# Funkcja do scrapowania danych z podanego linku
def scrape_data(link):
    response = requests.get(link)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        elements = soup.find_all(class_='css-1o924a9')
        return [element.get_text(strip=True) for element in elements]
    else:
        return []

# Obliczanie outlierów dla segmentu niskiego, odchylających się ceną w dół
low_segment = filtered_data[filtered_data['Segment'] == 'niski']
iqr_multiplier = 1.5

while True:
    q1_low = low_segment['Cena'].quantile(0.25)
    q3_low = low_segment['Cena'].quantile(0.75)
    iqr_low = q3_low - q1_low
    lower_bound_low = q1_low - iqr_multiplier * iqr_low

    # Filtrowanie outlierów odchylających się ceną w dół
    low_segment_outliers = low_segment[low_segment['Cena'] < lower_bound_low]

    if len(low_segment_outliers) >= 5 or iqr_multiplier <= 0:
        break

    iqr_multiplier -= 0.1

low_segment_outliers_df = pd.DataFrame(low_segment_outliers)

# Scrapowanie treści ogłoszeń dla outlierów w segmencie niskim
low_segment_outliers_df['Treść'] = low_segment_outliers_df['Link'].apply(scrape_data)

client = openai.OpenAI(api_key=api_key)


low_segment_outliers_df['Ocena_liczbowa_1do5'] = np.nan

def extract_search_terms(url_parts):
    terms = []
    for part in url_parts:
        match = re.search(r'q-([a-zA-Z0-9\-]+)', part)
        if match:
            # Zamiana "-" na spacje, rozdzielanie i dodanie do listy
            terms.extend(match.group(1).replace('-', ' ').split())
    return terms

search_terms = extract_search_terms(url_parts)
search_item = ' '.join(search_terms)

prompt = f"Mam wyselekcjonowne oferty {search_item} z OLX. Oto najlepsze z nich: {low_segment_outliers_df}, uzupełnij  dane z oceną liczbowa od 1 do 5. Wpisz ocenę stanu {search_item} w skali od 1 do 5 (jeśli przedmiot nie dotyczy wyszukiwanego przedmiotu, lub tylko jest akcesorium do {search_item} to wpisz 0). pisz tylko indeks i cyfra która jest oceną, odpowiedź w csv."

# Użyj klienta do stworzenia zapytania
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Only response, short as possible, no dot on end"},
        {"role": "user", "content": prompt}
    ]
)

response_AI = response.choices[0].message.content.strip()

# Czyszczenie danych z response_AI
cleaned_response_AI = response_AI.replace('```', '').strip()

# Przekształcenie response_AI do DataFrame
response_data = StringIO(cleaned_response_AI)
response_df = pd.read_csv(response_data, sep=",", header=None, names=['Index', 'Ocena_liczbowa_1do5'])

# Konwersja kolumny 'Index' na typ integer
response_df['Index'] = pd.to_numeric(response_df['Index'], errors='coerce').astype(int)

# Ustawienie indeksu na kolumnę 'Index'
response_df.set_index('Index', inplace=True)

# Konwersja indeksu response_df na typ indeksu low_segment_outliers_df
try:
    response_df.index = response_df.index.astype(low_segment_outliers_df.index.dtype)
except ValueError as e:
    st.error(f"Konwersja indeksu nie powiodła się: {e}")
    st.stop()

# Usuń istniejącą kolumnę 'Ocena_liczbowa_1do5' przed dołączeniem nowych danych, jeśli istnieje
if 'Ocena_liczbowa_1do5' in low_segment_outliers_df.columns:
    low_segment_outliers_df = low_segment_outliers_df.drop(columns=['Ocena_liczbowa_1do5'])

# Dołączenie ocen do low_segment_outliers_df
low_segment_outliers_df = low_segment_outliers_df.join(response_df, on=low_segment_outliers_df.index)

best_offerts = low_segment_outliers_df[low_segment_outliers_df['Ocena_liczbowa_1do5'] != 0]
best_offerts = best_offerts.drop(columns=['Segment'])

# Tworzenie nagłówków
table_string = " | ".join(best_offerts.columns) + "\n"
table_string += "-" * len(table_string) + "\n"

# Dodanie wierszy tabeli
for _, row in best_offerts.iterrows():
    table_string += " | ".join(map(str, row)) + "\n"

prompt1 = f"Mam tabelę która zawiera statystyki na temat ofer {search_item} z OLX. Przeanalizuj krótko tabelę: {summary_stats} , odpowiedz jaki jest najbardziej korzystny zakres w którym można kupić przedmiot. Następnie przeanalizuj tabelę z kilkoma najlepszymi ofertami:{table_string} , opisz która z tych obecych w tabeli jest najlepsza i daj tabelkę z nazwą oferty, ceną, oceną i czy do negocjacji i linkiem takim jak w tabeli, bez tworzenia odnośnika. Tabelka ma być ładnie sformatowana. Zwróć uwagę na kolumnę Treść która zawiera opis z ogłoszenia i Tekst która zawiera tytuł ogłoszenia, porównaj opisy, który opis sugeruje najlepszy przedmiot? może warto na coś zwrócić uwagę? dodatki do zakupu? Dodaj też że przyglądamy się tym ofertom najbardziej korzystnym cenowo dla {search_item}. Na koniec podkreśl i napisz pogrubieniem która oferta z tabeli jest najlepsza."

# Użyj klienta do stworzenia zapytania
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Ma to być informacja doradzająca jaki przedmiot wybrać, pamiętaj że użytkownik ma podaną tabelę ze statystykami, nie trzeba przywoływać ich w tekście"},
        {"role": "user", "content": prompt1}
    ],
)

response_AI = response.choices[0].message.content.strip()

st.markdown(response_AI)








