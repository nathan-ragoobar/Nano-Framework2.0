import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the style for better visualization
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# Load the data
data = pd.read_csv('gpt2_training.csv')

# Convert NA values to NaN
data = data.replace('NA', np.nan)

# Convert columns to appropriate types
numeric_cols = ['train_loss', 'val_loss', 'time_ms', 'tokens_per_second', 'learning_rate']
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col])

# Create a figure with subplots - 5 rows, 1 column
fig, axes = plt.subplots(5, 1, figsize=(14, 25), sharex=True)
fig.suptitle('GPT2 Training Metrics', fontsize=20)

# Plot 1: Training Loss
axes[0].plot(data['Step'].unique(), 
             data.groupby('Step')['train_loss'].mean().values, 
             marker='o', markersize=3, linestyle='-', color='blue', alpha=0.7)
axes[0].set_ylabel('Training Loss')
axes[0].set_title('Training Loss vs Step')

# Plot 2: Validation Loss
valid_data = data[~data['val_loss'].isna()]
if not valid_data.empty:
    axes[1].plot(valid_data['Step'].unique(), 
                 valid_data.groupby('Step')['val_loss'].mean().values, 
                 marker='o', markersize=3, linestyle='-', color='green', alpha=0.7)
axes[1].set_ylabel('Validation Loss')
axes[1].set_title('Validation Loss vs Step')

# Plot 3: Time (ms)
axes[2].plot(data['Step'].unique(), 
             data.groupby('Step')['time_ms'].mean().values, 
             marker='o', markersize=3, linestyle='-', color='red', alpha=0.7)
axes[2].set_ylabel('Time (ms)')
axes[2].set_title('Processing Time vs Step')

# Plot 4: Tokens per Second
axes[3].plot(data['Step'].unique(), 
             data.groupby('Step')['tokens_per_second'].mean().values, 
             marker='o', markersize=3, linestyle='-', color='purple', alpha=0.7)
axes[3].set_ylabel('Tokens per Second')
axes[3].set_title('Tokens per Second vs Step')

# Plot 5: Learning Rate
axes[4].plot(data['Step'].unique(), 
             data.groupby('Step')['learning_rate'].mean().values, 
             marker='o', markersize=3, linestyle='-', color='orange', alpha=0.7)
axes[4].set_ylabel('Learning Rate')
axes[4].set_title('Learning Rate vs Step')
axes[4].set_xlabel('Step')

# Adjusting layout and saving
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('gpt2_training_metrics.png', dpi=300, bbox_inches='tight')
plt.show()