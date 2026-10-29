import numpy as np
import pandas as pd

pd.set_option("display.width", 200)

 
# PART A : DATAFRAME OPERATIONS

dataframe1 = pd.DataFrame({
    "ProductID":  [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "NameofProd": ["Laptop", "Mouse", "Keyboard", "Monitor", "Printer",
                   "Scanner", "Webcam", "Headset", "Router", "Tablet"],
    "Price":      [55000, 500, 1200, 9000, 7500, 6500, 2500, 3000, 2200, 18000],
})

dataframe2 = pd.DataFrame({
    "ProductID":        [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
    "Company_Location": ["Chennai", "Mumbai", "Delhi", "Pune", "Bengaluru",
                         "Hyderabad", "Kolkata", "Jaipur", "Kochi", "Noida"],
})

dataframe3 = pd.DataFrame({
    "ProductID":  [111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
    "NameofProd": ["Speaker", "Pen Drive", "Hard Disk", "SSD", "Projector",
                   "Smart Watch", "Power Bank", "Charger", "HDMI Cable", "UPS"],
    "Price":      [1800, 600, 4200, 5200, 32000, 8000, 1500, 900, 350, 4800],
})

print("dataframe1\n", dataframe1, "\n")
print("dataframe2\n", dataframe2, "\n")
print("dataframe3\n", dataframe3, "\n")

# 1. Append the records of dataframe1 and dataframe3 (row-wise)
appended = pd.concat([dataframe1, dataframe3], ignore_index=True)
print("1. APPEND dataframe1 + dataframe3 (20 rows)\n", appended, "\n")

# 2. Concatenate wherever applicable
concat_rows = pd.concat([dataframe1, dataframe3], axis=0, ignore_index=True)
concat_cols = pd.concat([dataframe1, dataframe2], axis=1)
print("2a. CONCAT rows (axis=0) df1 & df3\n", concat_rows.head(), "\n")
print("2b. CONCAT columns (axis=1) df1 & df2\n", concat_cols, "\n")

# 3. Merge dataframe1 and dataframe2 using ProductID
merged = pd.merge(dataframe1, dataframe2, on="ProductID")
print("3. MERGE on ProductID\n", merged, "\n")

# 4. Join operations between dataframe1 and dataframe2
inner_join = pd.merge(dataframe1, dataframe2, on="ProductID", how="inner")
left_join = pd.merge(dataframe1, dataframe2, on="ProductID", how="left")
right_join = pd.merge(dataframe1, dataframe2, on="ProductID", how="right")
outer_join = pd.merge(dataframe1, dataframe2, on="ProductID", how="outer")

print("4a. INNER JOIN\n", inner_join, "\n")
print("4b. LEFT JOIN\n", left_join, "\n")
print("4c. RIGHT JOIN\n", right_join, "\n")
print("4d. FULL OUTER JOIN\n", outer_join, "\n")

# PART B : DATA TRANSFORMATIONS

REG_LAST4 = 287                       
Yuktha = np.array([REG_LAST4, 45, 12, 78, 33, 90, 21, 67, 54, 8], dtype=float)

print("Vector 'Yuktha':", Yuktha, "\n")


def rescale(x, new_min, new_max):
    return new_min + (x - x.min()) * (new_max - new_min) / (x.max() - x.min())


# 1. Normalisation to given ranges
norm_m3_p3 = rescale(Yuktha, -3, 3)
norm_m1_p1 = rescale(Yuktha, -1, 1)
norm_0_1 = rescale(Yuktha, 0, 1)          # classic Min-Max normalisation

print("1a. Normalised to [-3, +3]:\n", np.round(norm_m3_p3, 4))
print("1b. Normalised to [-1, +1]:\n", np.round(norm_m1_p1, 4))
print("1c. Min-Max [0, 1]:\n", np.round(norm_0_1, 4), "\n")

# 2. Z-score (sample standard deviation, same as R's scale())
z_score = (Yuktha - Yuktha.mean()) / Yuktha.std(ddof=1)
print("2. Z-score:\n", np.round(z_score, 4), "\n")

# 3. Decimal scaling:  v' = v / 10^j,  j = smallest integer with max(|v'|) < 1
j = int(np.floor(np.log10(np.abs(Yuktha).max()))) + 1
decimal_scaled = Yuktha / (10 ** j)
print(f"3. Decimal scaling (j = {j}):\n", np.round(decimal_scaled, 4))
