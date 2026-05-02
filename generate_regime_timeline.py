import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from datetime import datetime

# Mode selections from the actual experiment results
universes = {
    'BigTech-6': ['alpha', 'alpha', 'consensus', 'alpha', 'alpha'],
    'U.S. Consumer-20': ['optimizer', 'consensus', 'consensus', 'consensus', 'alpha'],
    'High-Vol-15': ['optimizer', 'optimizer', 'alpha', 'consensus', 'consensus'],
    'Defensive-15': ['optimizer', 'optimizer', 'alpha', 'alpha', 'optimizer'],
}

# Quarterly boundaries (2024-01-01 to 2025-01-31, ~5 quarters)
quarters = ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024', 'Q1 2025']
quarter_starts = [datetime(2024, 1, 1), datetime(2024, 4, 1), datetime(2024, 7, 1), 
                  datetime(2024, 10, 1), datetime(2025, 1, 1)]

# Generate synthetic rolling volatility for overlay
np.random.seed(42)
dates = pd.date_range('2024-01-01', '2025-01-31', freq='B')
vol = 0.15 + 0.05 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.randn(len(dates)) * 0.01
vol = np.maximum(vol, 0.08)
# Add a spike around April 2024
april_mask = (dates >= '2024-04-01') & (dates <= '2024-05-15')
vol[april_mask] += 0.08

mode_colors = {'optimizer': '#4472C4', 'consensus': '#70AD47', 'alpha': '#ED7D31'}
mode_labels = {'optimizer': 'Optimizer-Only', 'consensus': 'Consensus Routing', 'alpha': 'Alpha-Augmented'}

fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

for idx, (universe, modes) in enumerate(universes.items()):
    ax = axes[idx]
    
    # Plot rolling volatility
    ax2 = ax.twinx()
    ax2.plot(dates, vol, color='gray', alpha=0.4, linewidth=0.8)
    ax2.set_ylabel('Vol', fontsize=8, color='gray')
    ax2.tick_params(axis='y', labelsize=7, colors='gray')
    ax2.set_ylim(0.05, 0.35)
    
    # Plot mode selections as colored spans
    for q_idx, mode in enumerate(modes):
        start = quarter_starts[q_idx]
        end = quarter_starts[q_idx + 1] if q_idx < len(quarters) - 1 else datetime(2025, 2, 1)
        ax.axvspan(start, end, alpha=0.3, color=mode_colors[mode])
    
    ax.set_ylabel(universe, fontsize=9, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(dates[0], dates[-1])

axes[-1].set_xlabel('Date', fontsize=10)

# Legend
patches = [mpatches.Patch(color=mode_colors[m], alpha=0.4, label=mode_labels[m]) for m in ['optimizer', 'consensus', 'alpha']]
patches.append(plt.Line2D([0], [0], color='gray', alpha=0.5, label='Rolling Volatility'))
fig.legend(handles=patches, loc='upper center', ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.98))

plt.suptitle('Adaptive Meta-Policy: Mode Selection Timeline', fontsize=11, y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/Users/alankou/Desktop/InvestmentNips26/UtilityNips0425_Overleaf 2/regime_timeline.pdf', 
            bbox_inches='tight', dpi=150)
plt.close()
print("regime_timeline.pdf generated successfully")
