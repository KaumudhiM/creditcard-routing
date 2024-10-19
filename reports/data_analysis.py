import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Constants
DATA_PATH = "data/raw/transactions.csv"
FEE_STRUCTURE = {
    'Moneycard': {'success': 5, 'fail': 2},
    'Goldcard': {'success': 10, 'fail': 5},
    'UK_Card': {'success': 3, 'fail': 1},
    'Simplecard': {'success': 1, 'fail': 0.5}
}
RETRY_CAP = 10

def load_data(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None

def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

def calculate_transaction_fee(row):
    psp = row['PSP']
    outcome = 'success' if row['success'] == 1 else 'fail'
    return FEE_STRUCTURE[psp][outcome]

def main():
    df = load_data(DATA_PATH)
    if df is None:
        return

    # Display basic information about the dataset
    print("Dataset Info:")
    df.info()

    print("Basic Statistics:")
    print(df.describe())

    # Convert 'tmsp' column to datetime
    df['tmsp'] = pd.to_datetime(df['tmsp'])

    # Calculate and display the overall transaction success rate
    success_rate = df['success'].mean() * 100
    print(f"Overall Transaction Success Rate: {success_rate:.2f}%")

    # Display the number of transactions by country and PSP
    print("Transactions by Country:")
    print(df['country'].value_counts())

    print("Transactions by PSP:")
    print(df['PSP'].value_counts())

    # Visualize the distribution of transaction amounts
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['amount'], bins=30, kde=True, ax=ax)
    ax.set(title='Distribution of Transaction Amounts', xlabel='Amount', ylabel='Frequency')
    save_plot(fig, 'reports/charts/reports/charts/transaction_amounts_distribution.png')

    # Visualize the success rate by PSP
    success_by_psp = df.groupby('PSP')['success'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    success_by_psp.plot(kind='bar', ax=ax)
    ax.set(title='Success Rate by PSP', xlabel='PSP', ylabel='Success Rate')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    save_plot(fig, 'reports/charts/success_rate_by_psp.png')
    print("Exploratory Data Analysis completed. Two visualizations have been saved: 'transaction_amounts_distribution.png' and 'success_rate_by_psp.png'.")

    # Perform correlation analysis for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    print("Correlation Matrix:")
    print(correlation_matrix['success'].sort_values(ascending=False))

    # Visualize the correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
    ax.set(title='Correlation Heatmap')
    save_plot(fig, 'reports/charts/correlation_heatmap.png')
    print("Correlation analysis completed. A heatmap has been saved as 'correlation_heatmap.png'.")

    # Analyze and visualize the success rate by different categories
    print("Success Rate Analysis:")
    for col in ['country', 'PSP', 'card', '3D_secured']:
        success_rate = df.groupby(col)['success'].mean().sort_values(ascending=False)
        print(f"Success Rate by {col}:")
        print(success_rate)

        fig, ax = plt.subplots(figsize=(10, 6))
        success_rate.plot(kind='bar', ax=ax)
        ax.set(title=f'Success Rate by {col}', xlabel=col, ylabel='Success Rate')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        save_plot(fig, f'success_rate_by_{col}.png')

    print("Success rate analysis completed. Visualizations have been saved for each category.")

    # Analyze the impact of transaction amount on success
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='success', y='amount', data=df, ax=ax)
    ax.set(title='Transaction Amount by Success', xlabel='Success', ylabel='Transaction Amount')
    save_plot(fig, 'reports/charts/reports/charts/transaction_amount_by_success.png')
    print("Analysis of 3D Secure and transaction amount completed. Visualizations have been saved.")

    # Apply the transaction fee calculation
    df['transaction_fee'] = df.apply(calculate_transaction_fee, axis=1)

    # Calculate retries-related features
    df = df.sort_values(['country', 'tmsp'])
    df['is_retry'] = (df['amount'].shift(1) == df['amount']) & (df['tmsp'] - df['tmsp'].shift(1) <= pd.Timedelta(minutes=1))
    df['retry_count'] = df.groupby(['amount', 'country']).cumcount()
    df['is_first_attempt'] = (df['retry_count'] == 0).astype(int)
    df['time_since_first_attempt'] = df.groupby(['amount', 'country'])['tmsp'].transform(lambda x: (x - x.min()).dt.total_seconds())
    df['retry_count_capped'] = np.where(df['retry_count'] > RETRY_CAP, RETRY_CAP, df['retry_count'])

    # Analyze retry patterns
    retry_success_rate = df.groupby('retry_count_capped')['success'].mean().reset_index()
    retry_success_rate.columns = ['retry_count', 'success_rate']

    psp_retry_success_rate = df.groupby(['PSP', 'retry_count_capped'])['success'].mean().reset_index()
    psp_retry_success_rate.columns = ['PSP', 'retry_count', 'success_rate']

    psp_retry_count = df.groupby(['PSP', 'retry_count_capped'])['success'].count().reset_index()
    psp_retry_count.columns = ['PSP', 'retry_count', 'retry_frequency']

    # Plot overall success rate by retry count
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='retry_count', y='success_rate', data=retry_success_rate, marker='o', ax=ax)
    ax.set(title='Overall Success Rate by Retry Count (Capped at 10)', xlabel='Retry Count', ylabel='Success Rate')
    ax.set_xticks(retry_success_rate['retry_count'].unique())
    ax.grid()
    save_plot(fig, 'reports/charts/reports/charts/success_rate_by_retry_count.png')

    # Plot PSP-specific success rate by retry count
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='retry_count', y='success_rate', hue='PSP', data=psp_retry_success_rate, marker='o', ax=ax)
    ax.set(title='PSP-specific Success Rate by Retry Count (Capped at 10)', xlabel='Retry Count', ylabel='Success Rate')
    ax.set_xticks(psp_retry_success_rate['retry_count'].unique())
    ax.grid()
    ax.legend(title='PSP')
    save_plot(fig, 'reports/charts/reports/charts/psp_success_rate_by_retry_count.png')

    # Plot retry frequency by PSP and retry count
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='retry_count', y='retry_frequency', hue='PSP', data=psp_retry_count, ax=ax)
    ax.set(title='Retry Frequency by PSP and Retry Count (Capped at 10)', xlabel='Retry Count', ylabel='Number of Retries')
    ax.set_xticks(psp_retry_count['retry_count'].unique())
    ax.grid()
    ax.legend(title='PSP')
    save_plot(fig, 'reports/charts/reports/charts/retry_freq_by_psp_and_retry_count.png')

    # Display statistics for detailed insights
    print("Overall Success Rate by Retry Count (Capped at 10):")
    print(retry_success_rate)

    print("\nPSP-specific Success Rate by Retry Count (Capped at 10):")
    print(psp_retry_success_rate)

    print("\nRetry Frequency by PSP and Retry Count (Capped at 10):")
    print(psp_retry_count)

if __name__ == "__main__":
    main()