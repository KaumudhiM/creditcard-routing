import pandas as pd

def convert_excel_to_csv(excel_path, csv_path):
    df = pd.read_excel(excel_path)
    
    df.to_csv(csv_path, index=False)
    print(f"Excel file {excel_path} has been converted to CSV file {csv_path}")

if __name__ == "__main__":
    excel_file = 'data/raw/PSP_Jan_Feb_2019.xlsx'
    csv_file = 'data/raw/transactions.csv'
    convert_excel_to_csv(excel_file, csv_file)