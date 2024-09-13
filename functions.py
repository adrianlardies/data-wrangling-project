import pandas as pd
import yfinance as yf  # Librería para descargar los datos de Yahoo Finance
import requests
from dotenv import load_dotenv  # Para cargar variables de entorno como la API key
import os
import matplotlib.pyplot as plt  # Importar la librería para gráficos
import seaborn as sns

# Definir las fechas clave de los eventos
covid_crash_date = pd.Timestamp('2020-03-23')  # Mínimo durante la crisis del COVID-19
post_covid_peak_date = pd.Timestamp('2021-11-10')  # Pico de Bitcoin en 2021
rate_hike_start_date = pd.Timestamp('2022-01-01')  # Inicio del aumento de las tasas en 2022
inflation_peak_2021_start = pd.Timestamp('2021-01-01')  # Inicio del aumento fuerte de la inflación en 2021
inflation_peak_2021_end = pd.Timestamp('2022-01-01')    # Inflación sigue alta hacia principios de 2022
bitcoin_peak_date = pd.Timestamp('2021-11-10')  # Pico de Bitcoin en 2021
bitcoin_drop_date = pd.Timestamp('2022-06-18')  # Caída importante en 2022

# Función para obtener los datos del VIX (índice de volatilidad implícita)
# Se descarga el historial de los últimos 10 años desde Yahoo Finance y se limpian las columnas no necesarias
def get_vix_data():
    vix = yf.Ticker("^VIX")
    df_vix = vix.history(period="10y")
    df_vix = df_vix.drop(columns=['Dividends', 'Stock Splits', 'Volume', 'Open', 'High', 'Low'])  # Eliminar columnas irrelevantes
    df_vix.rename(columns={'Close': 'vix'}, inplace=True)  # Renombrar columna Close a 'vix'
    df_vix.reset_index(inplace=True)  # Convertir el índice en una columna
    df_vix['Date'] = pd.to_datetime(df_vix['Date'])  # Asegurar que la columna Date sea del tipo datetime
    df_vix['vix'] = df_vix['vix'].round(2)  # Redondear los valores de 'vix' a 2 decimales
    
    return df_vix[['Date', 'vix']]  # Devolver solo la fecha y el valor del VIX

# Función para obtener los datos de la tasa de interés de la Reserva Federal (IRX)
# Se obtienen los últimos 10 años de datos y se eliminan las columnas innecesarias
def get_interest_data():
    fed_rate = yf.Ticker("^IRX")
    df_interest = fed_rate.history(period="10y")
    df_interest = df_interest.drop(columns=['Open', 'High', 'Low', 'Dividends', 'Stock Splits', 'Volume'])
    df_interest.rename(columns={'Close': 'interest_rate'}, inplace=True)  # Renombrar columna Close a 'interest_rate'
    df_interest.reset_index(inplace=True)
    df_interest['Date'] = pd.to_datetime(df_interest['Date'])  # Asegurar que la columna Date sea datetime
    df_interest['interest_rate'] = df_interest['interest_rate'].round(2)  # Redondear a 2 decimales
    
    return df_interest[['Date', 'interest_rate']]  # Devolver solo la fecha y la tasa de interés

# Función para obtener los datos del CPI (Índice de Precios al Consumidor) e inflación
# Se usa la API del Banco de Luisiana (FRED) para descargar los datos y se interpolan para obtener valores diarios
def get_cpi_inflation_data():
    load_dotenv()  # Cargar variables de entorno desde .env para obtener la API key
    api_key = os.getenv('API_BANK_INFLATION')  # Obtener la API key del entorno
    series_id = 'CPIAUCNS'  # Series ID para el CPI (Índice de Precios al Consumidor)
    
    # Solicitar los datos a la API de FRED
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json'
    response = requests.get(url)
    data = response.json()

    # Convertir los datos a DataFrame
    observations = data['observations']
    df_cpi = pd.DataFrame(observations)
    df_cpi.drop(columns=['realtime_start', 'realtime_end'], inplace=True)  # Eliminar columnas innecesarias
    df_cpi['date'] = pd.to_datetime(df_cpi['date'])  # Convertir la columna date a datetime
    df_cpi['value'] = pd.to_numeric(df_cpi['value'], errors='coerce')  # Asegurar que los valores sean numéricos
    df_cpi = df_cpi.dropna(subset=['value'])  # Eliminar valores nulos
    df_cpi.set_index('date', inplace=True)
    df_cpi = df_cpi[df_cpi.index >= '2010-01-01']  # Filtrar datos desde 2010

    # Resample para obtener el valor anual y calcular la inflación
    df_annual = df_cpi.resample('A').last()  # Obtener el último valor de cada año
    df_annual['inflation'] = df_annual['value'].pct_change() * 100  # Calcular inflación anual
    df_annual.rename(columns={'value': 'cpi'}, inplace=True)

    # Interpolar los valores anuales a diarios
    df_daily_cpi = df_annual[['cpi']].resample('D').interpolate(method='linear')  # Interpolación lineal para obtener valores diarios
    df_daily_cpi['inflation'] = df_annual['inflation'].resample('D').ffill()  # Rellenar con forward fill los valores de inflación

    # Rellenar valores NaN y redondear a dos decimales
    df_daily_cpi = df_daily_cpi.fillna(method='ffill').fillna(method='bfill').round(2)
    
    return df_daily_cpi.reset_index()  # Devolver el dataframe con el CPI y la inflación diarios

# Función para limpiar y combinar los datos de oro, bitcoin, S&P 500, VIX, tasa de interés y CPI/inflación
# Se asegura que todas las fechas estén estandarizadas y se combinan los dataframes por fecha
def clean_and_combine(df_gold, df_bitcoin, df_sp500, df_vix, df_interest, df_cpi_inflation):
    # Asegurar que todas las fechas sean datetime y eliminar zonas horarias donde sea necesario
    df_gold['Date'] = pd.to_datetime(df_gold['Date'])
    df_bitcoin['Date'] = pd.to_datetime(df_bitcoin['Date'])
    df_sp500['DATE'] = pd.to_datetime(df_sp500['DATE'])
    df_vix['Date'] = pd.to_datetime(df_vix['Date']).dt.tz_localize(None)  # Eliminar zona horaria del VIX
    df_interest['Date'] = pd.to_datetime(df_interest['Date']).dt.tz_localize(None)  # Eliminar zona horaria de tasa de interés
    df_cpi_inflation['date'] = pd.to_datetime(df_cpi_inflation['date'])

    # Renombrar y limpiar los dataframes de oro y bitcoin
    df_gold = df_gold[['Date', 'Price', 'Change %']].rename(columns={'Date': 'date', 'Price': 'price_gold', 'Change %': 'change_gold'})
    df_gold['price_gold'] = pd.to_numeric(df_gold['price_gold'].str.replace(',', ''), errors='coerce').round(0).astype(int)
    df_gold['change_gold'] = df_gold['change_gold'].str.replace('%', '').astype(float)

    df_bitcoin = df_bitcoin[['Date', 'Price', 'Change %']].rename(columns={'Date': 'date', 'Price': 'price_bitcoin', 'Change %': 'change_bitcoin'})
    df_bitcoin['price_bitcoin'] = pd.to_numeric(df_bitcoin['price_bitcoin'].str.replace(',', ''), errors='coerce').round(0).astype(int)
    df_bitcoin['change_bitcoin'] = df_bitcoin['change_bitcoin'].str.replace('%', '').astype(float)

        # Limpiar el dataframe del S&P 500
    df_sp500 = df_sp500.rename(columns={'DATE': 'date', 'SP500': 'price_sp500'})
    df_sp500['price_sp500'] = pd.to_numeric(df_sp500['price_sp500'].str.replace(',', ''), errors='coerce').fillna(method='ffill').round(0).astype(int)

    # Renombrar las columnas de 'Date' a 'date' en los dataframes de VIX y tasa de interés
    df_vix = df_vix.rename(columns={'Date': 'date'})
    df_interest = df_interest.rename(columns={'Date': 'date'})

    # Unir los dataframes por la columna 'date'
    # Se hace una serie de merges usando 'left' para asegurar que todas las fechas importantes estén presentes
    df_combined = pd.merge(df_gold, df_bitcoin, on='date', how='inner')
    df_combined = pd.merge(df_combined, df_sp500, on='date', how='inner')
    df_combined = pd.merge(df_combined, df_vix, on='date', how='left')
    df_combined = pd.merge(df_combined, df_interest, on='date', how='left')
    df_combined = pd.merge(df_combined, df_cpi_inflation, on='date', how='left')

    # Reordenar las columnas del dataframe combinado para un formato más claro y legible
    df_combined = df_combined[['date', 'price_bitcoin', 'price_gold', 'price_sp500', 'change_bitcoin', 'change_gold', 'vix', 'interest_rate', 'cpi', 'inflation']]

    return df_combined  # Devolver el dataframe final combinado

def calculate_investment_evolution(df):
    # Asegurarse de que la columna 'date' esté en formato datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Filtrar el último día de cada mes. Resample toma solo la última fila de cada mes
    # Esto nos permite trabajar con precios mensuales en lugar de diarios
    df_monthly = df.resample('M', on='date').last()

    # Definir una inversión inicial de $100 en cada activo (Bitcoin, Oro, S&P 500)
    initial_investment = 100
    
    # Crear nuevas columnas para almacenar la evolución de la inversión de cada activo
    df_monthly['investment_bitcoin'] = initial_investment
    df_monthly['investment_gold'] = initial_investment
    df_monthly['investment_sp500'] = initial_investment
    
    # Iterar a través de cada mes (a partir del segundo mes) para calcular la evolución de la inversión
    for i in range(1, len(df_monthly)):
        # Cálculo de la nueva inversión en Bitcoin:
        # La nueva inversión es igual a la inversión anterior multiplicada por la variación porcentual en el precio,
        # más la nueva inversión de $100
        df_monthly.loc[df_monthly.index[i], 'investment_bitcoin'] = (
            df_monthly.loc[df_monthly.index[i - 1], 'investment_bitcoin'] *  # Inversión anterior
            (df_monthly.loc[df_monthly.index[i], 'price_bitcoin'] / df_monthly.loc[df_monthly.index[i - 1], 'price_bitcoin']) +  # Variación porcentual
            initial_investment  # Nueva inversión de $100
        )
        
        # Cálculo de la nueva inversión en Oro con la misma lógica que Bitcoin
        df_monthly.loc[df_monthly.index[i], 'investment_gold'] = (
            df_monthly.loc[df_monthly.index[i - 1], 'investment_gold'] *
            (df_monthly.loc[df_monthly.index[i], 'price_gold'] / df_monthly.loc[df_monthly.index[i - 1], 'price_gold']) +
            initial_investment
        )
        
        # Cálculo de la nueva inversión en S&P 500 con la misma lógica que Bitcoin y Oro
        df_monthly.loc[df_monthly.index[i], 'investment_sp500'] = (
            df_monthly.loc[df_monthly.index[i - 1], 'investment_sp500'] *
            (df_monthly.loc[df_monthly.index[i], 'price_sp500'] / df_monthly.loc[df_monthly.index[i - 1], 'price_sp500']) +
            initial_investment
        )
    
    # Ahora, volvemos a unir las columnas de inversión calculadas al dataframe original (con fechas diarias)
    # Usamos la fusión basada en el periodo mensual ('M') de la columna de fechas
    df = pd.merge(
        df,  # Dataframe original con datos diarios
        df_monthly[['investment_bitcoin', 'investment_gold', 'investment_sp500']],  # Dataframe mensual con la evolución de inversión
        left_on=df['date'].dt.to_period('M'),  # Convertir las fechas diarias a periodos mensuales para alinear los datos
        right_on=df_monthly.index.to_period('M'),  # Usar el índice mensual de df_monthly
        how='left'  # Usar un merge a la izquierda para mantener todas las filas del dataframe original
    )
    
    # Eliminar la columna 'key_0' que se genera durante el merge y que no es necesaria
    df.drop(columns='key_0', inplace=True)

    # Retornar el dataframe final con la evolución de las inversiones añadida
    return df

# Función para generar el gráfico de precios con escala logarítmica y anotaciones
def plot_evolution_of_prices_with_events(df_combined):
    plt.figure(figsize=(12, 8))

    # Trazar las líneas de los precios
    plt.plot(df_combined['date'], df_combined['price_bitcoin'], label='Bitcoin', color='orange')
    plt.plot(df_combined['date'], df_combined['price_gold'], label='Oro', color='gold')
    plt.plot(df_combined['date'], df_combined['price_sp500'], label='S&P 500', color='blue')

    # Añadir escala logarítmica en el eje Y
    plt.yscale('log')

    # Eventos importantes de Bitcoin
    plt.axvline(pd.Timestamp('2016-07-09'), color='gray', linestyle='--', linewidth=1)
    plt.text(pd.Timestamp('2016-07-09'), 100, 'Bitcoin Halving (2016)', rotation=90, verticalalignment='bottom')

    plt.axvline(pd.Timestamp('2020-05-11'), color='gray', linestyle='--', linewidth=1)
    plt.text(pd.Timestamp('2020-05-11'), 100, 'Bitcoin Halving (2020)', rotation=90, verticalalignment='bottom')

    plt.axvline(pd.Timestamp('2021-02-08'), color='gray', linestyle='--', linewidth=1)
    plt.text(pd.Timestamp('2021-02-08'), 100, 'Tesla compra Bitcoin', rotation=90, verticalalignment='bottom')

    plt.axvline(pd.Timestamp('2021-05-12'), color='gray', linestyle='--', linewidth=1)
    plt.text(pd.Timestamp('2021-05-12'), 100, 'Tesla deja de aceptar Bitcoin', rotation=90, verticalalignment='bottom')

    # Eventos importantes del oro
    plt.axvline(pd.Timestamp('2020-08-01'), color='gray', linestyle='--', linewidth=1)
    plt.text(pd.Timestamp('2020-08-01'), 4000, 'Máximo histórico del oro (2020)', rotation=90, verticalalignment='bottom')

    # Eventos importantes del S&P 500
    plt.axvline(pd.Timestamp('2020-03-01'), color='gray', linestyle='--', linewidth=1)
    plt.text(pd.Timestamp('2020-03-01'), 400, 'Crisis por COVID-19', rotation=90, verticalalignment='bottom')

    # Títulos y etiquetas
    plt.title('Evolución de los Precios: Bitcoin, Oro y S&P 500 (Escala Logarítmica) con Eventos Clave')
    plt.xlabel('Fecha')
    plt.ylabel('Precio ($)')
    plt.xlim(pd.Timestamp('2015-01-01'), pd.Timestamp('2024-12-31'))  # Limitar el eje X entre 2015 y 2024
    plt.legend()
    plt.grid(True)

    # Mostrar el gráfico
    plt.show()

    # Función para obtener el valor de la inversión en las fechas clave más cercanas
def get_closest_investment_value(df, target_date):
    closest_row = df.iloc[(df['date'] - target_date).abs().argmin()]
    investment_bitcoin = closest_row['investment_bitcoin']
    investment_gold = closest_row['investment_gold']
    investment_sp500 = closest_row['investment_sp500']
    actual_date = closest_row['date']
    return investment_bitcoin, investment_gold, investment_sp500, actual_date

# Función para graficar la evolución de la inversión con fechas clave
def plot_investment_evolution_with_key_dates(df_combined):
    # Fechas clave
    start_date = pd.Timestamp('2015-01-01')
    bitcoin_peak_date = pd.Timestamp('2021-04-14')  # Pico de Bitcoin en abril de 2021
    end_date = pd.Timestamp('2024-02-09')  # Fecha final en el DataFrame

    # Obtener los valores de inversión más cercanos a las fechas clave
    investment_start = get_closest_investment_value(df_combined, start_date)
    investment_bitcoin_peak = get_closest_investment_value(df_combined, bitcoin_peak_date)
    investment_end = get_closest_investment_value(df_combined, end_date)

    # Crear gráfico con escala logarítmica para la inversión en Bitcoin, Oro y S&P 500
    plt.figure(figsize=(10, 6))

    # Graficar las líneas de inversión
    plt.plot(df_combined['date'], df_combined['investment_bitcoin'], label='Inversión en Bitcoin', color='orange')
    plt.plot(df_combined['date'], df_combined['investment_gold'], label='Inversión en Oro', color='gold')
    plt.plot(df_combined['date'], df_combined['investment_sp500'], label='Inversión en S&P 500', color='blue')

    # Añadir escala logarítmica en el eje Y
    plt.yscale('log')

    # Añadir anotaciones de fechas clave
    plt.axvline(start_date, color='gray', linestyle='--', linewidth=1)
    plt.text(start_date, investment_start[0], 'Inicio (2015)', rotation=90, verticalalignment='bottom')

    plt.axvline(bitcoin_peak_date, color='gray', linestyle='--', linewidth=1)
    plt.text(bitcoin_peak_date, investment_bitcoin_peak[0], 'Pico de Bitcoin (2021)', rotation=90, verticalalignment='bottom')

    plt.axvline(end_date, color='gray', linestyle='--', linewidth=1)
    plt.text(end_date, investment_end[0], 'Final (2024)', rotation=90, verticalalignment='bottom')

    # Títulos y etiquetas
    plt.title('Evolución de una Inversión de $100 mensuales en Bitcoin, Oro y S&P 500 (Escala Logarítmica)')
    plt.xlabel('Fecha')
    plt.ylabel('Valor de la inversión ($)')
    plt.legend()
    plt.grid(True)

    # Mostrar el gráfico
    plt.show()

# Función para graficar el VIX junto con Bitcoin, Oro y S&P 500
def plot_vix_vs_prices(df_combined):
    # Crear gráfico mejorado con VIX, Bitcoin, Oro y S&P 500
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Graficar VIX
    ax1.plot(df_combined['date'], df_combined['vix'], color='green', label='VIX')

    # Graficar Bitcoin y Oro
    ax2.plot(df_combined['date'], df_combined['price_bitcoin'], color='orange', label='Bitcoin')
    ax2.plot(df_combined['date'], df_combined['price_gold'], color='gold', label='Oro')

    # Añadir S&P 500
    ax2.plot(df_combined['date'], df_combined['price_sp500'], color='blue', label='S&P 500')

    # Añadir etiquetas y leyendas
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('VIX', color='green')
    ax2.set_ylabel('Precio ($)', color='blue')
    ax1.tick_params(axis='y', labelcolor='green')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Añadir título y leyenda
    plt.title('VIX (Volatilidad) vs Precio de Bitcoin, Oro y S&P 500')
    ax2.legend(loc='upper left')

    # Mostrar gráfico
    plt.grid(True)
    plt.show()

# Función para encontrar los picos más altos del VIX
def get_vix_peak_dates(df, n=3):
    return df.nlargest(n, 'vix')['date'].values

# Función para calcular los cambios porcentuales en los precios antes y después del evento
def calculate_percentage_change(df, event_date, column, window_days=30):
    # Filtrar los datos dentro de la ventana antes y después del evento
    before_event = df[(df['date'] >= event_date - pd.Timedelta(days=window_days)) & (df['date'] < event_date)]
    after_event = df[(df['date'] > event_date) & (df['date'] <= event_date + pd.Timedelta(days=window_days))]
    
    # Obtener el valor antes y después del evento
    value_before = before_event[column].iloc[-1] if not before_event.empty else None
    value_after = after_event[column].iloc[0] if not after_event.empty else None
    
    if value_before is not None and value_after is not None:
        # Calcular el cambio porcentual
        change_percentage = ((value_after - value_before) / value_before) * 100
        return change_percentage, value_before, value_after
    return None, None, None

# Función para analizar los picos del VIX y calcular los cambios porcentuales de precios
def analyze_vix_peaks_and_price_changes(df_combined, window_days=30):
    # Encontrar los picos más altos del VIX
    vix_peak_dates = get_vix_peak_dates(df_combined)
    
    # Calcular los cambios porcentuales en los precios de Bitcoin, Oro y S&P 500
    price_changes = {}
    for event_date in vix_peak_dates:
        price_changes[str(event_date)] = {
            'Bitcoin': calculate_percentage_change(df_combined, pd.Timestamp(event_date), 'price_bitcoin', window_days),
            'Oro': calculate_percentage_change(df_combined, pd.Timestamp(event_date), 'price_gold', window_days),
            'S&P 500': calculate_percentage_change(df_combined, pd.Timestamp(event_date), 'price_sp500', window_days)
        }
    
    # Retornar los resultados
    return price_changes

# Función para graficar la tasa de interés junto con Bitcoin, Oro y S&P 500
def plot_interest_vs_prices(df_combined):
    # Crear gráfico que muestra la tasa de interés, Bitcoin, Oro y S&P 500
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Graficar tasa de interés en el eje izquierdo
    ax1.plot(df_combined['date'], df_combined['interest_rate'], color='red', label='Tasa de interés')

    # Graficar Bitcoin, Oro y S&P 500 en el eje derecho
    ax2.plot(df_combined['date'], df_combined['price_bitcoin'], color='orange', label='Bitcoin')
    ax2.plot(df_combined['date'], df_combined['price_gold'], color='gold', label='Oro')
    ax2.plot(df_combined['date'], df_combined['price_sp500'], color='blue', label='S&P 500')

    # Añadir etiquetas y leyendas
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Tasa de interés (%)', color='red')
    ax2.set_ylabel('Precio ($)', color='blue')
    ax1.tick_params(axis='y', labelcolor='red')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Añadir título y leyenda
    plt.title('Tasa de Interés vs Precios de Bitcoin, Oro y S&P 500')
    ax2.legend(loc='upper left')

    # Añadir grid
    plt.grid(True)

    # Mostrar gráfico
    plt.show()

# Función para encontrar las fechas más cercanas a los eventos clave
def get_closest_value(df, target_date, column):
    closest_row = df.iloc[(df['date'] - target_date).abs().argmin()]
    return closest_row[column], closest_row['date']

# Función para calcular el cambio porcentual entre dos fechas clave
def calculate_percentage_change_closest(df, start_date, end_date, column):
    before_event, actual_start_date = get_closest_value(df, start_date, column)
    after_event, actual_end_date = get_closest_value(df, end_date, column)
    change_percentage = ((after_event - before_event) / before_event) * 100
    return change_percentage, before_event, after_event, actual_start_date, actual_end_date

# Función para analizar cambios porcentuales en eventos clave
def analyze_event_price_changes(df_combined):
    # Calcular los cambios porcentuales durante los eventos clave de COVID-19 y el aumento de tasas
    price_changes_covid = {
        'Bitcoin': calculate_percentage_change_closest(df_combined, covid_crash_date, post_covid_peak_date, 'price_bitcoin'),
        'Oro': calculate_percentage_change_closest(df_combined, covid_crash_date, post_covid_peak_date, 'price_gold'),
        'S&P 500': calculate_percentage_change_closest(df_combined, covid_crash_date, post_covid_peak_date, 'price_sp500')
    }

    price_changes_rate_hike = {
        'Bitcoin': calculate_percentage_change_closest(df_combined, post_covid_peak_date, rate_hike_start_date, 'price_bitcoin'),
        'Oro': calculate_percentage_change_closest(df_combined, post_covid_peak_date, rate_hike_start_date, 'price_gold'),
        'S&P 500': calculate_percentage_change_closest(df_combined, post_covid_peak_date, rate_hike_start_date, 'price_sp500')
    }

    # Retornar los resultados
    return price_changes_covid, price_changes_rate_hike

# Función para graficar la inflación junto con Bitcoin, Oro y S&P 500
def plot_inflation_comparison(df_combined):
    # Crear gráfico mejorado que compara la inflación con los precios de Bitcoin, Oro y S&P 500
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Graficar inflación en el eje izquierdo
    ax1.plot(df_combined['date'], df_combined['inflation'], color='red', linestyle='--', label='Inflación')

    # Graficar Bitcoin, Oro y S&P 500 en el eje derecho
    ax2.plot(df_combined['date'], df_combined['price_bitcoin'], color='orange', label='Bitcoin')
    ax2.plot(df_combined['date'], df_combined['price_gold'], color='gold', label='Oro')
    ax2.plot(df_combined['date'], df_combined['price_sp500'], color='blue', label='S&P 500')

    # Añadir etiquetas y leyendas
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Inflación (%)', color='red')
    ax2.set_ylabel('Precio ($)', color='blue')
    ax1.tick_params(axis='y', labelcolor='red')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Añadir título y leyenda
    plt.title('Comparación del Precio de Bitcoin, Oro y S&P 500 con la Inflación')
    ax2.legend(loc='upper left')

    # Añadir grid
    plt.grid(True)

    # Mostrar gráfico
    plt.show()

# Función para encontrar el valor más cercano a una fecha en particular
def get_closest_value(df, target_date, column):
    closest_row = df.iloc[(df['date'] - target_date).abs().argmin()]
    return closest_row[column], closest_row['date']

# Función para calcular los cambios porcentuales en los precios antes y después del evento
def calculate_percentage_change_period(df, start_date, end_date, column):
    before_event, actual_start_date = get_closest_value(df, start_date, column)
    after_event, actual_end_date = get_closest_value(df, end_date, column)
    change_percentage = ((after_event - before_event) / before_event) * 100
    return change_percentage, before_event, after_event, actual_start_date, actual_end_date

# Función para analizar cambios de precios durante el pico de inflación (2021-2022)
def analyze_inflation_peak_price_changes(df_combined):
    # Calcular los cambios porcentuales para Bitcoin, Oro y S&P 500 durante el periodo de inflación alta (2021-2022)
    price_changes_inflation_2021 = {
        'Bitcoin': calculate_percentage_change_period(df_combined, inflation_peak_2021_start, inflation_peak_2021_end, 'price_bitcoin'),
        'Oro': calculate_percentage_change_period(df_combined, inflation_peak_2021_start, inflation_peak_2021_end, 'price_gold'),
        'S&P 500': calculate_percentage_change_period(df_combined, inflation_peak_2021_start, inflation_peak_2021_end, 'price_sp500')
    }

    return price_changes_inflation_2021

def plot_correlation_heatmap(df_combined):
    # Reordenar las columnas para una visualización más lógica
    ordered_columns = ['price_bitcoin', 'price_gold', 'price_sp500', 'inflation', 'interest_rate', 'vix']
    corr_matrix_ordered = df_combined[ordered_columns].corr()

    # Crear una nueva figura para el mapa de calor mejorado
    plt.figure(figsize=(10, 6))

    # Crear el mapa de calor mejorado con anotaciones más claras
    sns.heatmap(corr_matrix_ordered, annot=True, cmap="coolwarm", vmin=-1, vmax=1, annot_kws={"size": 10}, linewidths=0.5, linecolor='gray')

    # Añadir título y ajustar etiquetas
    plt.title('Mapa de Calor de Correlación entre Variables (Mejorado)', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Mostrar el gráfico
    plt.show()

# Función para crear un gráfico de área apilada de la inversión en Bitcoin, Oro y S&P 500
def plot_stacked_investment_growth(df_combined):
    # Crear gráfico de área apilada mejorado con transparencia y anotaciones clave
    plt.figure(figsize=(10, 6))

    # Añadir transparencia a las áreas para que se puedan ver mejor las contribuciones
    plt.stackplot(df_combined['date'], 
                  df_combined['investment_bitcoin'], 
                  df_combined['investment_gold'], 
                  df_combined['investment_sp500'], 
                  labels=['Bitcoin', 'Oro', 'S&P 500'], 
                  colors=['orange', 'gold', 'blue'], alpha=0.8)

    # Añadir títulos y etiquetas
    plt.title('Evolución Acumulada de la Inversión en Bitcoin, Oro y S&P 500 (Mejorado)')
    plt.xlabel('Fecha')
    plt.ylabel('Valor de la Inversión ($)')

    # Añadir leyenda
    plt.legend(loc='upper left')

    # Añadir grid
    plt.grid(True)

    # Mostrar gráfico
    plt.show()

# Función para calcular el valor total de la cartera en una fecha clave
def calculate_portfolio_composition(df, date):
    # Filtrar la fila más cercana a la fecha proporcionada
    closest_row = df.iloc[(df['date'] - date).abs().argmin()]
    total_investment = (closest_row['investment_bitcoin'] +
                        closest_row['investment_gold'] +
                        closest_row['investment_sp500'])
    
    # Calcular la proporción de cada activo
    bitcoin_percentage = (closest_row['investment_bitcoin'] / total_investment) * 100
    gold_percentage = (closest_row['investment_gold'] / total_investment) * 100
    sp500_percentage = (closest_row['investment_sp500'] / total_investment) * 100
    
    return {
        'Bitcoin': bitcoin_percentage,
        'Oro': gold_percentage,
        'S&P 500': sp500_percentage,
        'Total ($)': total_investment
    }

# Función para calcular las proporciones de la cartera en fechas clave
def analyze_portfolio_composition(df_combined):
    # Calcular las proporciones de la cartera en el pico de Bitcoin y la caída en 2022
    composition_at_peak = calculate_portfolio_composition(df_combined, bitcoin_peak_date)
    composition_at_drop = calculate_portfolio_composition(df_combined, bitcoin_drop_date)
    
    return composition_at_peak, composition_at_drop

# Función para crear el gráfico de dispersión con burbujas (Inflación vs Precio)
def plot_bubble_inflation_vs_price(df_combined):
    # Crear gráfico de dispersión con burbujas mejorado
    plt.figure(figsize=(10, 6))

    # Graficar Bitcoin
    plt.scatter(df_combined['inflation'], df_combined['price_bitcoin'], 
                s=df_combined['vix']*10, c='orange', alpha=0.5, label='Bitcoin', edgecolor='black', linewidth=1)

    # Graficar Oro
    plt.scatter(df_combined['inflation'], df_combined['price_gold'], 
                s=df_combined['vix']*10, c='gold', alpha=0.5, label='Oro', edgecolor='black', linewidth=1)

    # Añadir títulos y etiquetas
    plt.title('Inflación vs Precio con Volatilidad (VIX) Representada por Tamaño (Mejorado)')
    plt.xlabel('Inflación (%)')
    plt.ylabel('Precio ($)')
    plt.legend(loc='upper right')

    # Añadir grid
    plt.grid(True)

    # Mostrar gráfico
    plt.show()

# Función para calcular los retornos y generar un gráfico de violín
def plot_violin_returns(df_combined):
    # Calcular los retornos de Bitcoin, Oro y S&P 500
    df_combined['bitcoin_return'] = df_combined['price_bitcoin'].pct_change() * 100
    df_combined['gold_return'] = df_combined['price_gold'].pct_change() * 100
    df_combined['sp500_return'] = df_combined['price_sp500'].pct_change() * 100

    # Crear gráfico de violín mejorado con colores diferenciados
    plt.figure(figsize=(10, 6))

    # Crear el gráfico de violín
    sns.violinplot(data=df_combined[['bitcoin_return', 'gold_return', 'sp500_return']], palette='muted')

    # Añadir título y etiquetas
    plt.title('Distribución de Retornos para Bitcoin, Oro y S&P 500 (Mejorado)')
    plt.ylabel('Retorno (%)')
    plt.xticks([0, 1, 2], ['Bitcoin', 'Oro', 'S&P 500'])

    # Añadir grid
    plt.grid(True)

    # Mostrar gráfico
    plt.show()

# Función para crear el gráfico de burbuja (Precio de Bitcoin vs Oro con VIX y Tasa de Interés)
def plot_bubble_interest_vix(df_combined):
    # Crear gráfico de burbuja mejorado ajustando los colores y tamaños para una mejor visualización
    plt.figure(figsize=(10, 6))

    # Graficar con mejor ajuste de colores y tamaños de burbujas
    scatter = plt.scatter(df_combined['price_bitcoin'], df_combined['price_gold'], 
                          s=df_combined['vix']*5, 
                          c=df_combined['interest_rate'], cmap='coolwarm', alpha=0.7)

    # Añadir la barra de color para las tasas de interés
    cbar = plt.colorbar(scatter)
    cbar.set_label('Tasa de Interés')

    # Añadir título y etiquetas
    plt.title('Precio de Bitcoin vs Oro, con VIX y Tasa de Interés (Mejorado)')
    plt.xlabel('Precio de Bitcoin ($)')
    plt.ylabel('Precio de Oro ($)')
    plt.grid(True)

    # Mostrar el gráfico mejorado
    plt.show()

# Función para calcular medias de precios de Bitcoin y Oro en momentos de altas tasas de interés y alta volatilidad
def analyze_high_interest_and_volatility(df_combined):
    # Definir umbrales clave para tasas de interés altas (>4%) y VIX alto (>30)
    interest_rate_threshold = 4  # Tasa de interés alta (>4%)
    vix_threshold = 30  # VIX alto (>30)

    # Filtrar datos con altas tasas de interés y alta volatilidad (VIX alto)
    high_interest_data = df_combined[df_combined['interest_rate'] > interest_rate_threshold]
    high_vix_data = df_combined[df_combined['vix'] > vix_threshold]

    # Calcular medias de precios de Bitcoin y Oro en momentos de altas tasas de interés y alta volatilidad
    bitcoin_mean_price_high_interest = high_interest_data['price_bitcoin'].mean()
    gold_mean_price_high_interest = high_interest_data['price_gold'].mean()

    bitcoin_mean_price_high_vix = high_vix_data['price_bitcoin'].mean()
    gold_mean_price_high_vix = high_vix_data['price_gold'].mean()

    # Retornar los resultados
    return {
        'Bitcoin (Tasa de interés > 4%)': bitcoin_mean_price_high_interest,
        'Oro (Tasa de interés > 4%)': gold_mean_price_high_interest,
        'Bitcoin (VIX > 30)': bitcoin_mean_price_high_vix,
        'Oro (VIX > 30)': gold_mean_price_high_vix
    }