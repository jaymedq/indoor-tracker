import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines

def create_positioning_illustration():
    # Academic styling
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
    })

    # 1. Set up the main figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    # ax.set_title("Comparison of GNSS and IPS Coverage in Urban Environments", pad=15)
    ax.axis('off') # Hide coordinate axes for a cleaner diagram

    # 2. Draw the Building (The Indoor Environment)
    # x=35, y=0, width=40, height=50
    building = patches.Rectangle((35, 0), 40, 50, linewidth=1.5, edgecolor='black', facecolor='white', hatch='///', zorder=2)
    ax.add_patch(building)
    
    # Draw the building floor
    ax.plot([0, 100], [0, 0], color='black', linewidth=3.0, zorder=3)
    
    # Environment Labels
    plt.text(55, 25, 'Indoor\nEnvironment', horizontalalignment='center', fontsize=12, zorder=3, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    plt.text(17, 10, 'Outdoor\nEnvironment', horizontalalignment='center', fontsize=12, zorder=3)
    plt.text(83, 10, 'Outdoor\nEnvironment', horizontalalignment='center', fontsize=12, zorder=3)

    # 3. Draw GNSS Satellites and Signals (Space to Ground)
    satellites = [(20, 90), (50, 95), (80, 90)]
    for sx, sy in satellites:
        # Draw the satellite
        ax.plot(sx, sy, marker='*', markersize=15, color='black', linestyle='None', zorder=4)
        plt.text(sx + 3, sy, 'GNSS Satellite', fontsize=10, verticalalignment='center')

        # Draw successful GNSS signals reaching the outdoor ground
        if sx < 35:
            ax.plot([sx, 15], [sy, 0], color='black', linestyle='--', linewidth=1.2, alpha=0.7)
            ax.plot([sx, 30], [sy, 0], color='black', linestyle='--', linewidth=1.2, alpha=0.7)
        elif sx > 75:
            ax.plot([sx, 80], [sy, 0], color='black', linestyle='--', linewidth=1.2, alpha=0.7)
            ax.plot([sx, 95], [sy, 0], color='black', linestyle='--', linewidth=1.2, alpha=0.7)

        # Draw blocked GNSS signals hitting the building roof
        blocked_x1, blocked_x2 = 45, 65
        ax.plot([sx, blocked_x1], [sy, 50], color='black', linestyle=':', linewidth=1.5, alpha=0.9)
        ax.plot([sx, blocked_x2], [sy, 50], color='black', linestyle=':', linewidth=1.5, alpha=0.9)
        
        # Add "X" markers where the signal is blocked
        ax.plot(blocked_x1, 50, marker='x', color='black', markersize=8, linestyle='None', zorder=5) 
        ax.plot(blocked_x2, 50, marker='x', color='black', markersize=8, linestyle='None', zorder=5)

    # 4. Draw Positioning Sensors and Coverage (Inside the Building)
    sensors = [(42, 40), (68, 40), (55, 12)]
    for bx, by in sensors:
        # Draw the sensor
        ax.plot(bx, by, marker='^', markersize=10, color='black', linestyle='None', zorder=5)
        
        # Draw the localized indoor coverage area
        coverage = patches.Circle((bx, by), 7, alpha=0.2, color='gray', zorder=4)
        ax.add_patch(coverage)

    # plt.text(55, 42, 'Positioning Sensors', horizontalalignment='center', color='black', fontsize=11, zorder=4, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))

    # 5. Create a Custom Legend
    gnss_satelite = mlines.Line2D([], [], color='black', linestyle='None', marker='*', markersize=10, label='GNSS Satellite')
    gnss_line = mlines.Line2D([], [], color='black', linestyle='--', linewidth=1.2, label='GNSS Signal (LOS)')
    blocked_line = mlines.Line2D([], [], color='black', linestyle=':', linewidth=1.5, marker='x', markersize=8, label='GNSS Signal (Blocked)')
    sensor_marker = mlines.Line2D([], [], color='black', linestyle='None', marker='^', markersize=10, label='Positioning Sensor')
    ips_patch = patches.Patch(color='gray', alpha=0.2, label='IPS Anchor Coverage')
    building_patch = patches.Patch(facecolor='white', edgecolor='black', hatch='///', label='Indoor Environment')
    
    # Put the legend below the plot
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.legend(handles=[gnss_satelite, gnss_line, blocked_line, building_patch, sensor_marker, ips_patch], loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=True, edgecolor='black', facecolor='white', framealpha=1, fancybox=False)

    # Render the plot
    plt.tight_layout()
    plt.savefig('gnss_vs_ips_academic.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('gnss_vs_ips_academic.png', format='png', bbox_inches='tight', dpi=300)
    # plt.show() # Uncomment to show interactivelly when running

if __name__ == "__main__":
    create_positioning_illustration()