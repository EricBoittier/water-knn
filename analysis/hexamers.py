import struct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
#  matplotlib styles
plt.style.use(["science", "no-latex", "ieee"])

markersize = 15



xticklabels = [
    'prism',
    'cage', 
    'book 1',
    'book 2',
    'bag',
    'cyclic chair',
    'cyclic boat 1',
    'cyclic boat 2'
]

def read_energies(log_path, affix=""):
    """Read energies from log files"""
    log_files = list(Path(log_path).glob("cc*log"+affix))
    assert len(log_files) == 8, f"Found {len(log_files)} log files"
    energies = []
    energies2 = []
    conformers = []
    
    for log_file in log_files:
        print(log_file)
        with open(log_file) as f:
            lines_ = [_ for _ in f.readlines() if _.__contains__("ENER EXTERN>")]
            
            # First energy
            spl = lines_[0].split()
            ans = float(spl[2]) + float(spl[3])
            energies.append(ans)
            
            # Second energy
            spl = lines_[1].split()
            ans = float(spl[2]) + float(spl[3])
            energies2.append(ans)
            
            conformers.append(str(log_file.stem)[6:-4])
            
    return pd.DataFrame({
        "E1": energies,
        "E2": energies2, 
        "conformer": conformers
    })

def get_conformer_categories():
    """Return ordered list of conformer categories"""
    return [
        'bag.xyz',
        'book1.xyz',
        'book2.xyz',
        'cyclic_boat1.xyz',
        'cyclic_boat2.xyz',
        'cyclic_chair.xyz',
        'prism.xyz',
        'cage.xyz'
    ]

def get_reference_energies():
    """Return dictionary of reference energies"""
    return {
        "ccsd(t)": [
            -46.8,   # bag
            -47.52,  # b1
            -47.26,  # b2
            -45.44,  # cb1
            -45.36,  # cb2
            -46.47,  # cchair
            -48.24,  # prism
            -47.95   # cage
        ],
        
        "ccsd(t)3b": [
            -35.28 - 10.35,  # bag
            -36.02 - 10.38,  # b1
            -36.13 - 10.11,  # b2
            -32.3 - 11.34,   # cb1
            -32.24 - 11.34,  # cb2
            -32.71 - 11.78,  # cchair
            -38.94 - 8.7,    # prism
            -38.47 - 8.97    # cage
        ],
        
        "ccsd(t)2b": [
            -35.28,  # bag
            -36.02,  # b1
            -36.13,  # b2
            -32.3,   # cb1
            -32.24,  # cb2
            -32.71,  # cchair
            -38.94,  # prism
            -38.47   # cage
        ],
        
        "mbpol": [
            -46.3,   # bag
            -47.00,  # b1
            -46.81,  # b2
            -44.76,  # cb1
            -44.76,  # cb2
            -45.63,  # cchair
            -48.17,  # prism
            -47.85   # cage
        ]
    }



def process_data(log_path, affix=""):
    # Read data and process
    test_df = read_energies(log_path, affix=affix)
    assert len(test_df) == 8, f"Found {len(test_df)} log files"
    print(test_df)
    # Set up categories
    cats = get_conformer_categories()
    test_df["cat"] = pd.Categorical(test_df["conformer"], categories=cats, ordered=True)
    test_df.sort_values("cat")
    print(test_df)

    # Create main dataframe
    ref_energies = get_reference_energies()
    data = pd.DataFrame({
        "ccsd(t)": ref_energies["ccsd(t)"],
        "ccsd(t)2b": ref_energies["ccsd(t)2b"],
        "ccsd(t)3b": ref_energies["ccsd(t)3b"],
        "mbpol": ref_energies["mbpol"],
        "conformer": cats
    })
    test_df["conformer"] = test_df["conformer"].str.replace(".xyz", "")
    test_df["conformer"] = test_df["conformer"].str.replace(".inp", "")
    data["conformer"] = data["conformer"].str.replace(".xyz", "")
    print("test_df")
    print(test_df)  
    data = data.sort_values("ccsd(t)")
    print("data")
    print(data)
    data = data.merge(test_df, on="conformer")

    assert len(data) == 8, f"Found {len(data)} log files"
    return data


# Global variables
N = 0

# data = process_data(log_path)

xticks = np.arange(0, len(xticklabels), 1)

colors = {
    "ccsd(t)": "goldenrod",
    "ccsd(t)2b": "goldenrod",
    "ccsd(t)3b": "goldenrod",
    "mbpol": "gray",

}




# Styles for data
def get_styles(label, color1="black", color2="black", markersize=10, linewidth=1):
    styles = {
        "ccsd(t)": {
        "color": colors["ccsd(t)"],
        "linestyle": "-",
        "marker": "o",
        "label": "CCSD(T)",
        "markersize": markersize,
        "linewidth": linewidth
    },
    "ccsd(t)2b": {
        "color": colors["ccsd(t)2b"],
        "linestyle": "--",
        "marker": "$\\rm 2B$",
        "markersize": markersize,
        "label": "CCSD(T)-2B",
        "linewidth": linewidth
    },
    "ccsd(t)3b": {
        "color": colors["ccsd(t)3b"],
        "linestyle": "-.",
        "marker": "$\\rm 3B$",
        "label": "CCSD(T)-3B",
        "markersize": markersize,
        "linewidth": linewidth
    },
    "mbpol": {
        "color": colors["mbpol"],
        "linestyle": "-",
        "marker": "P",
        "label": "MB-pol",
        "markersize": markersize,
        "linewidth": linewidth
    },
    "E1": {
        "color": color1,
        "linestyle": "-",
        "marker": "h",
        # "label": "kMDCM-NN",
        "markersize": markersize,
        "linewidth": linewidth
    },
    "E2": {
        "color": color2,
        "linestyle": "--",
        "marker": "*",
        # "label": "kMDCM-NN" + " (opt.)",
        "markersize": markersize,
        "linewidth": linewidth
    }
    }
    return styles


def get_relative_energies(_):
    data = _.copy()
    """Get relative energies"""
    for key in data.keys():
        if key not in ["conformer", "cat"]:
            data[key] = data[key] - data[key].min()
    return data




# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=False)


E_ylim = (-49, -32)
E_ylim_relative = (-0.1, 7)

affix = ""
log_path = Path("/home/boittier/pcbach/waterlj/kparms-4.0-d594b332-5427-4d8a-9b79-64f1ac10ed1a")

styles = get_styles("kMDCM-NN"+affix, color1="b", color2="b")
data = process_data(log_path, affix=affix)

print(data)


rmse_dft = np.sqrt(np.mean((data["E1"] - data["ccsd(t)"])**2))
rmse_dftopt = np.sqrt(np.mean((data["E2"] - data["ccsd(t)"])**2))


relative_data = get_relative_energies(data)



# plot data
for key, value in styles.items():
    axes[0, 0].plot(data["conformer"], data[key], **value)
    axes[0, 0].set_ylim(E_ylim)

# plot relative data
for key, value in styles.items():
    axes[0, 1].plot(relative_data["conformer"], relative_data[key], **value)
    axes[0,1].set_ylim(E_ylim_relative)


# load m-cc data
log_path = "/home/boittier/pcbach/kmdcmmdimersccsdt"
affix = ".1"

styles = get_styles("kMDCM-NN"+affix, color1="b", color2="b")

data = process_data(log_path, affix=affix)

rmse_ccsdt = np.sqrt(np.mean((data["E1"] - data["ccsd(t)"]))**2)
rmse_ccsdtopt = np.sqrt(np.mean((data["E2"] - data["ccsd(t)"]))**2)

print("data")
print(data)
print(rmse_ccsdt, rmse_ccsdtopt)

relative_data = get_relative_energies(data)


# plot m-cc data
for key, value in styles.items():
    axes[1, 0].plot(data["conformer"], data[key], **value)
    axes[1, 0].set_xticks(xticks)   
    axes[1, 0].set_xticklabels(xticklabels, rotation=45)
    axes[1, 0].set_ylim(E_ylim)

# plot m-cc relative data
for key, value in styles.items():
    axes[1, 1].plot(relative_data["conformer"], relative_data[key], **value)
    axes[1, 1].set_xticks(xticks)
    axes[1, 1].set_xticklabels(xticklabels, rotation=45)
    axes[1, 1].set_ylim(E_ylim_relative)

# legend to the left outside of the plot
axes[1,1].legend(loc="center left", bbox_to_anchor=(0.01, 0.75))

#axis labels on left columm 
energylabel = "$\Delta E$\n(kcal/mol)"
relativeenergylabel = "$\Delta E - \Delta E_{min}$\n(kcal/mol)"

axes[0, 0].set_ylabel(energylabel)
axes[0, 1].set_ylabel(relativeenergylabel)
axes[1, 0].set_ylabel(energylabel)
axes[1, 1].set_ylabel(relativeenergylabel)

axes_list = ["A1", "A2", "B1", "B2"]

for ax, label in zip(axes.flatten(), axes_list):
    ax.text(1 - 0.15, 1 - 0.9, label, transform=ax.transAxes, fontsize=12, fontweight="bold")


rmse_ccsdt = np.sqrt(np.mean((data["E1"] - data["ccsd(t)"])**2))
rmse_ccsdtopt = np.sqrt(np.mean((data["E2"] - data["ccsd(t)"])**2))

# axes[0, 0].text(1.1, 1.1, "$\omega$B97X-D", transform=axes[0, 0].transAxes, fontsize=12, fontweight="bold")
for i in range(2):
    label = "M-DFT" if i == 0 else "M-CC"
    for j in range(2):
        axes[i, j].text(0.01, 0.938,
                 label, transform=axes[i, j].transAxes, fontsize=8, fontweight="bold", color="b", alpha=0.5)
        # axes[1, 0].text(1.1, 1.1, "$\omega$B97X-D", transform=axes[1, 0].transAxes, fontsize=12, fontweight="bold")
        # axes[i, j].text(0.01, 0.938, 
        #                 "M-CC", 
        #                 transform=axes[1, 0].transAxes, fontsize=8, fontweight="bold", color="b", alpha=0.5)


# axes[0, 0].text(0.01, 0.938,
#                  "M-DFT RMSE: {:.1f}/{:.1f} kcal/mol".format(rmse_dft, rmse_dftopt), transform=axes[0, 0].transAxes, fontsize=8, fontweight="bold", color="b", alpha=0.5)
# # axes[1, 0].text(1.1, 1.1, "$\omega$B97X-D", transform=axes[1, 0].transAxes, fontsize=12, fontweight="bold")
# axes[1, 0].text(0.01, 0.938, 
#                 "M-CC RMSE: {:.1f}/{:.1f} kcal/mol".format(rmse_ccsdt, rmse_ccsdtopt), 
#                 transform=axes[1, 0].transAxes, fontsize=8, fontweight="bold", color="b", alpha=0.5)


# adjust spacing between subplots
plt.subplots_adjust(wspace=0.2, hspace=0.05)

plt.savefig("pngs/hexamers.png", dpi=300)
plt.savefig("pdfs/hexamers.pdf", dpi=300)
