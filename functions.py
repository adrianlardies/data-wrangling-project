import pandas as pd
import yfinance as yf  # Librería para descargar los datos de Yahoo Finance
import requests
from dotenv import load_dotenv  # Para cargar variables de entorno como la API key
import os
import matplotlib.pyplot as plt  # Importar la librería para gráficos

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