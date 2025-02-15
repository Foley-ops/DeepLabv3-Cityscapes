import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv('VOS_benchmark_results.csv', parse_dates=['Timestamp'])
df.set_index('Timestamp', inplace=True)

# Choose columns to plot (skip if missing on some machines)
columns_to_plot = []
for col in ['Avg Inference Time (s)', 'Avg CPU Usage (%)', 'Avg Memory Usage (%)', 'Avg GPU Power (W)']:
    if col in df.columns:
        columns_to_plot.append(col)

# 1. Line plots over time
df[columns_to_plot].plot(subplots=True, figsize=(8, 6), title=columns_to_plot)
plt.tight_layout()
plt.show()

# 2. Scatter plots (correlations)
# Example correlation: GPU Power vs Inference Time
if 'Avg GPU Power (W)' in df.columns and 'Avg Inference Time (s)' in df.columns:
    df.plot.scatter(x='Avg GPU Power (W)', y='Avg Inference Time (s)', title='GPU Power vs Inference Time')
    plt.show()

# Example correlation: CPU Usage vs Inference Time
if 'Avg CPU Usage (%)' in df.columns and 'Avg Inference Time (s)' in df.columns:
    df.plot.scatter(x='Avg CPU Usage (%)', y='Avg Inference Time (s)', title='CPU Usage vs Inference Time')
    plt.show()
