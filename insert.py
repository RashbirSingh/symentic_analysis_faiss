import pandas as pd
from sqlalchemy import create_engine


def insert_data(file_name, table_name):
    db_user = 'admin'
    db_password = 'admin123'
    db_host = 'localhost'
    db_port = '5432'
    db_name = 'symentic_analysis'

    df = pd.read_csv(file_name)
    if file_name == 'complaints.csv':
        df.rename(columns={'Unnamed: 0': 'Complaint_No'}, inplace=True)

    engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

    df.to_sql(table_name, engine, if_exists='append', index=False)

    print("CSV data inserted successfully!")

    engine.dispose()