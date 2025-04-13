import pandas as pd

def solana_data_cleaning():
    df = pd.read_csv('solana_data.csv')  # Make sure this path matches your file

    # Convert 'Time of Scraping' to datetime format
    df['Time of Scraping'] = pd.to_datetime(df['Time of Scraping'])

    # Function to clean dollar signs, commas, and percentage signs
    def clean_numeric(col):
        return col.replace({r'\$': '', ',': '', '%': ''}, regex=True).astype(float)

    # Clean all numeric columns
    for col in ['Price', 'Market Cap', '24h Trading Volume', 'Circulating Supply',
                '1h Price Change %', '24h Price Change %', '7d Price Change %']:
        df[col] = clean_numeric(df[col])

    # Extract the 24h High and Low from the range column
    range_split = df['24h Range (High/Low)'].str.extract(r'\$(?P<Low>[\d,.]+)\s*-\s*\$(?P<High>[\d,.]+)')
    df['24h Low'] = range_split['Low'].str.replace(',', '').astype(float)
    df['24h High'] = range_split['High'].str.replace(',', '').astype(float)
    df['24h Avg Price'] = df[['24h Low', '24h High']].mean(axis=1)

    # Calculate price change compared to previous record
    df['Price Change Since Last'] = df['Price'].diff()

    # Select only the final cleaned columns
    df_cleaned = df[['Time of Scraping', 'Price', 'Price Change Since Last', '24h Low', '24h High', '24h Avg Price',
                     'Market Cap', '24h Trading Volume', 'Circulating Supply',
                     '1h Price Change %', '24h Price Change %', '7d Price Change %']]

    # Save to CSV
    df_cleaned.to_csv('Solana_cleaned.csv', index=False)
    print("✅ Solana data cleaned and saved to 'data/Solana_cleaned.csv'!")

# Run the cleaning
solana_data_cleaning()
