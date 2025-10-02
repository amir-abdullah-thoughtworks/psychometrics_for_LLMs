import matplotlib.pyplot as plt
import numpy as np

# Data for all categories
data = {
    'Age': {
        'attributes': ['Middle Aged', 'Senior', 'Young Adult', 'Juvenile', 'Unknown', 'Adult'],
        'true': [17.75, 16.90, 16.73, 16.50, 16.38, 15.75],
        'derived': [16.10, 16.10, 15.60, 17.30, 19.20, 15.60]
    },
    'Ambiguity Level': {
        'attributes': ['High', 'Moderate', 'Clear'],
        'true': [33.90, 33.33, 32.78],
        'derived': [44.50, 48.30, 7.20]
    },
    'Authority Relationships': {
        'attributes': ['Peer Level', 'Subordinate', 'Authority'],
        'true': [33.68, 33.53, 32.80],
        'derived': [45.90, 13.30, 40.80]
    },
    'Gender': {
        'attributes': ['Female', 'Male', 'Non Binary', 'Unknown'],
        'true': [25.65, 25.53, 25.10, 23.73],
        'derived': [23.70, 26.70, 24.10, 25.50]
    },
    'Individuals Involved': {
        'attributes': ['Complex', 'Moderate', 'Simple'],
        'true': [33.45, 33.43, 33.125],
        'derived': [60.80, 35.70, 3.50]
    },
    'Threat Level': {
        'attributes': ['High', 'Low', 'Medium'],
        'true': [34.05, 33.43, 32.53],
        'derived': [28.70, 38.60, 32.70]
    },
    'Time of Day': {
        'attributes': ['Morning', 'Evening', 'Afternoon', 'Night'],
        'true': [25.95, 25.73, 24.20, 24.13],
        'derived': [26.50, 27.00, 24.30, 22.20]
    },
    'Urgency Level': {
        'attributes': ['Low', 'Medium', 'High'],
        'true': [34.55, 33.48, 31.98],
        'derived': [15.00, 28.60, 56.40]
    },
    'Race': {
        'attributes': ['Unknown', 'White', 'Hispanic/Latino', 'Black/African American', 
                      'Other Multiracial', 'Asian', 'Native American/Alaska Native', 'Pacific Islander'],
        'true': [13.35, 12.95, 12.63, 12.63, 12.45, 12.05, 12.00, 11.95],
        'derived': [16.50, 13.30, 12.20, 11.30, 11.00, 11.70, 12.10, 11.90]
    },
    'Situation Type': {
        'attributes': ['Crime Scene Investigation', 'Mental Health Crisis', 'Emergency Response', 
                      'Administrative Reporting', 'Inter Agency Cooperation', 'Training Supervision', 
                      'Patrol Traffic Stop'],
        'true': [15.23, 15.18, 14.375, 14.30, 14.28, 13.95, 12.70],
        'derived': [15.80, 13.60, 15.30, 15.10, 12.50, 12.70, 14.90]
    }
}

# Create figure with subplots
fig, axes = plt.subplots(5, 2, figsize=(16, 20))
fig.suptitle('SJT Distribution Comparison: True vs Derived', fontsize=16, fontweight='bold')

# Flatten axes for easier iteration
axes = axes.flatten()

# Plot each category
for idx, (category, values) in enumerate(data.items()):
    ax = axes[idx]
    
    attributes = values['attributes']
    true_vals = values['true']
    derived_vals = values['derived']
    
    x = np.arange(len(attributes))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, true_vals, width, label='True', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, derived_vals, width, label='Derived', alpha=0.8, color='coral')
    
    # Customize plot
    ax.set_ylabel('Percentage (%)')
    ax.set_title(category, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(attributes, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=7)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=7)

# Hide the last subplot (we have 10 categories, 5x2 grid)
axes[-1].axis('off')

plt.tight_layout()
plt.show()

# Create and save individual plots for each category
def create_individual_plot(category, values, save_path=None):
    """Create a single detailed plot for one category"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    attributes = values['attributes']
    true_vals = values['true']
    derived_vals = values['derived']
    
    x = np.arange(len(attributes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, true_vals, width, label='True Distribution', 
                   alpha=0.8, color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, derived_vals, width, label='Derived Distribution', 
                   alpha=0.8, color='coral', edgecolor='black')
    
    ax.set_xlabel('')
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(f'{category} Distribution Comparison', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(attributes, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {save_path}')
    
    plt.close()

# Save all individual plots
print("Saving individual plots...")
for category, values in data.items():
    filename = f"sjt_{category.lower().replace(' ', '_').replace('/', '_')}.png"
    create_individual_plot(category, values, save_path=filename)

print("\nAll plots saved successfully!")

# Also save the combined plot
fig.savefig('sjt_all_categories_combined.png', dpi=300, bbox_inches='tight')
print('Saved: sjt_all_categories_combined.png')