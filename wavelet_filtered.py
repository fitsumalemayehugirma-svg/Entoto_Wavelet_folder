import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt

# SIMPLE GEOMAGNETIC WAVELET FILTERING
# ENTX, ENTY, ENTZ and ENTF



# 1. SETTINGS

WAVELET = "haar"
LEVEL = 6

# Filter D6, D5 and D4
TARGET_LEVELS = [1, 2, 3]

COMPONENTS = [
    "ENTX",
    "ENTY",
    "ENTZ",
    "ENTF"
]

OUTPUT = "Wavelet_Results"

os.makedirs(OUTPUT, exist_ok=True)

os.makedirs(
    os.path.join(OUTPUT, "Daily_Data"),
    exist_ok=True
)

os.makedirs(
    os.path.join(OUTPUT, "Daily_Plots"),
    exist_ok=True
)

os.makedirs(
    os.path.join(OUTPUT, "Daily_Tables"),
    exist_ok=True
)


 
# 2. FIND .MIN FILES
 

files = sorted(
    glob.glob("*.min")
)

if len(files) == 0:

    print("ERROR: No .min files found.")

    print(
        "Put your .min files in the same folder "
        "as this Python program."
    )

    raise SystemExit


print("\nFiles found:")

for f in files:
    print("  ", f)


 
# 3. READ MAGNETIC FILE
 

def read_file(filename):

    with open(filename, "r") as f:

        lines = f.readlines()


    header = None

    for i, line in enumerate(lines):

        if line.startswith("DATE"):

            header = i
            break


    if header is None:

        raise ValueError(
            "Could not find DATE header."
        )


    data = pd.read_csv(
        filename,
        sep=r"\s+",
        skiprows=header
    )


    # Create time column

    data["datetime"] = pd.to_datetime(
        data["DATE"].astype(str)
        + " "
        + data["TIME"].astype(str)
    )


    return data


 
# 4. WAVELET FILTER
 
def wavelet_filter(signal):

    # DWT decomposition

    coefficients = pywt.wavedec(
        signal,
        WAVELET,
        level=LEVEL
    )


    filtered = coefficients.copy()


    # Filter selected levels

    for i in TARGET_LEVELS:

        c = filtered[i]


        # Estimate noise

        sigma = (
            np.median(
                np.abs(
                    c - np.median(c)
                )
            )
            / 0.6745
        )


        # Threshold

        threshold = 3 * sigma


        # Soft threshold

        filtered[i] = pywt.threshold(
            c,
            threshold,
            mode="soft"
        )


    # Reconstruct signal

    result = pywt.waverec(
        filtered,
        WAVELET
    )


    # Keep original length

    result = result[
        :len(signal)
    ]


    return result


 
# 5. PROCESS EACH DAY
 

all_days = []


for file in files:

    print("\n")
    print("=" * 60)
    print("PROCESSING:", file)
    print("=" * 60)


    # Read file

    df = read_file(file)


    # Remove missing values

    df = df.dropna(
        subset=COMPONENTS
    ).copy()


    # Date

    date = (
        df["datetime"]
        .iloc[0]
        .strftime("%Y-%m-%d")
    )


    print(
        "Date:",
        date
    )

    print(
        "Number of measurements:",
        len(df)
    )


     
    # Create output dataframe
    

    result = df[
        [
            "DATE",
            "TIME",
            "datetime"
        ]
        + COMPONENTS
    ].copy()


    
    # Filter all four components
  

    for component in COMPONENTS:

        print(
            "Filtering:",
            component
        )


        signal = df[
            component
        ].values.astype(float)


        filtered = wavelet_filter(
            signal
        )


        result[
            component + "_filtered"
        ] = filtered


     
    # SAVE COMPLETE DAILY DATA
     
    daily_file = os.path.join(
        OUTPUT,
        "Daily_Data",
        f"{date}_filtered.csv"
    )


    result.to_csv(
        daily_file,
        index=False
    )


    print(
        "Saved:",
        daily_file
    )


     
    # DAILY GRAPH
     
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(15, 12),
        sharex=True
    )


    for ax, component in zip(
        axes,
        COMPONENTS
    ):

        ax.plot(
            result["datetime"],
            result[component],
            linewidth=0.8,
            label="Original"
        )


        ax.plot(
            result["datetime"],
            result[
                component + "_filtered"
            ],
            linewidth=1.5,
            label="Wavelet Filtered"
        )


        ax.set_ylabel(
            component + "\n(nT)",
            fontsize=10
        )


        ax.grid(
            True,
            alpha=0.25,
            linestyle="--"
        )


        ax.legend(
            fontsize=9
        )


        ax.spines[
            "top"
        ].set_visible(False)


        ax.spines[
            "right"
        ].set_visible(False)


    axes[-1].set_xlabel(
        "Time",
        fontsize=12
    )


    fig.suptitle(
        f"Entoto Geomagnetic Field\n"
        f"Wavelet Filtering — {date}",
        fontsize=18,
        fontweight="bold"
    )


    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96
        ]
    )


    graph_file = os.path.join(
        OUTPUT,
        "Daily_Plots",
        f"{date}_wavelet_filtering.png"
    )


    plt.savefig(
        graph_file,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()

    plt.close()

 
    # DAILY SAMPLE TABLE
    # ========================================================

    # Take first 20 measurements

    table = result.head(20).copy()


    # Keep useful columns

    table_columns = [
        "datetime"
    ]


    for component in COMPONENTS:

        table_columns.append(
            component
        )

        table_columns.append(
            component + "_filtered"
        )


    table = table[
        table_columns
    ]


    # Round numbers

    for column in table.columns:

        if column != "datetime":

            table[column] = table[
                column
            ].round(3)


    # Save table

    table_file = os.path.join(
        OUTPUT,
        "Daily_Tables",
        f"{date}_sample_table.csv"
    )


    table.to_csv(
        table_file,
        index=False
    )


    print(
        "Sample table saved:",
        table_file
    )


    # Display table

    print("\nSample of filtered data:")

    print(
        table.to_string(
            index=False
        )
    )


    # Store day

    all_days.append(
        result
    )



# 6. COMBINE ALL DAYS
 
complete = pd.concat(
    all_days,
    ignore_index=True
)


complete = complete.sort_values(
    "datetime"
).reset_index(
    drop=True
)


 
# 7. SAVE COMPLETE FIVE-DAY DATA
 

complete_file = os.path.join(
    OUTPUT,
    "Complete_5_Day_Filtered.csv"
)


complete.to_csv(
    complete_file,
    index=False
)


print("\n")
print(
    "Complete five-day data saved:"
)

print(
    complete_file
)


 
# 8. FIVE-DAY GRAPH
 

fig, axes = plt.subplots(
    4,
    1,
    figsize=(16, 12),
    sharex=True
)


for ax, component in zip(
    axes,
    COMPONENTS
):

    ax.plot(
        complete["datetime"],
        complete[component],
        linewidth=0.7,
        label="Original"
    )


    ax.plot(
        complete["datetime"],
        complete[
            component + "_filtered"
        ],
        linewidth=1.3,
        label="Wavelet Filtered"
    )


    ax.set_ylabel(
        component + "\n(nT)",
        fontsize=10
    )


    ax.grid(
        True,
        alpha=0.25,
        linestyle="--"
    )


    ax.legend(
        fontsize=9
    )


    ax.spines[
        "top"
    ].set_visible(False)


    ax.spines[
        "right"
    ].set_visible(False)


axes[-1].set_xlabel(
    "Date and Time",
    fontsize=12
)


fig.suptitle(
    "Entoto Geomagnetic Field\n"
    "Complete Five-Day Wavelet Filtering",
    fontsize=18,
    fontweight="bold"
)


plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.96
    ]
)


five_day_graph = os.path.join(
    OUTPUT,
    "Complete_5_Day_Comparison.png"
)


plt.savefig(
    five_day_graph,
    dpi=300,
    bbox_inches="tight"
)


plt.show()

plt.close()


 
# 9. FIVE-DAY SUMMARY TABLE
 
summary = []


for component in COMPONENTS:

    original = complete[
        component
    ].values


    filtered = complete[
        component + "_filtered"
    ].values


    original_std = np.std(
        original
    )


    filtered_std = np.std(
        filtered
    )


    rmse = np.sqrt(
        np.mean(
            (
                original
                - filtered
            ) ** 2
        )
    )


    summary.append(
        {
            "Component": component,
            "Original_SD_nT":
                round(original_std, 3),

            "Filtered_SD_nT":
                round(filtered_std, 3),

            "RMSE_nT":
                round(rmse, 3)
        }
    )


summary_df = pd.DataFrame(
    summary
)


summary_file = os.path.join(
    OUTPUT,
    "Five_Day_Summary_Table.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


print("\n")
print("=" * 60)
print("FIVE-DAY SUMMARY")
print("=" * 60)

print(
    summary_df.to_string(
        index=False
    )
)


 
# 10. FINISHED
 

print("\n")
print("=" * 60)
print("PROJECT COMPLETED")
print("=" * 60)

print()
print("Wavelet method:")
print("  Haar DWT")
print("  6 decomposition levels")
print("  D4-D6 soft thresholding")
print("  IDWT reconstruction")

print()
print("Components:")
print("  ENTX")
print("  ENTY")
print("  ENTZ")
print("  ENTF")

print()
print("Results saved in:")
print(
    OUTPUT
)

print()
print("Finished!")
