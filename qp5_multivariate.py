import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates

sns.set_theme(style="whitegrid")
URL = "https://raw.githubusercontent.com/salemprakash/EDA/main/Data/Employee%20Attrition.csv"

# 1. Load + identify numerical / categorical variables
df = pd.read_csv(URL)
numerical = df.select_dtypes(include="number").columns.tolist()
categorical = df.select_dtypes(exclude="number").columns.tolist()
print("1. Shape:", df.shape)
print("   Numerical variables  :", numerical)
print("   Categorical variables:", categorical, "\n")

# 2. Scatter plot - 3 variables (X, Y, colour)
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x="Age", y="MonthlyIncome", hue="Attrition", alpha=0.7, palette="Set1")
plt.title("2. Age vs MonthlyIncome coloured by Attrition")
plt.tight_layout(); plt.show()

# 3. Scatter plot - 4 variables (X, Y, colour, shape)
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x="Age", y="MonthlyIncome", hue="Attrition", style="Gender",
                alpha=0.7, palette="Set1")
plt.title("3. Age vs MonthlyIncome: colour = Attrition, shape = Gender")
plt.tight_layout(); plt.show()

# 4. Scatter plot - 4 variables (X, Y, colour, size = numeric)
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x="Age", y="MonthlyIncome", hue="Attrition", size="YearsAtCompany",
                sizes=(20, 250), alpha=0.6, palette="Set1")
plt.title("4. Age vs MonthlyIncome: colour = Attrition, size = YearsAtCompany")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout(); plt.show()

# 5. Bubble plot - 3 numerical variables
sample = df.sample(250, random_state=1)           # sample keeps bubbles readable
plt.figure(figsize=(9, 6))
plt.scatter(sample["Age"], sample["MonthlyIncome"], s=sample["TotalWorkingYears"] * 12,
            alpha=0.5, color="dodgerblue", edgecolor="black")
plt.xlabel("Age"); plt.ylabel("MonthlyIncome")
plt.title("5. Bubble Plot: size = TotalWorkingYears")
plt.tight_layout(); plt.show()

# 6. facet_wrap equivalent
g = sns.relplot(data=df, x="Age", y="MonthlyIncome", col="JobRole", col_wrap=3,
                hue="Attrition", height=3, alpha=0.7, palette="Set1")
g.figure.suptitle("6. facet_wrap: Age vs MonthlyIncome by JobRole", y=1.02)
plt.show()

# 7. facet_grid equivalent (rows = Gender, columns = Department)
g = sns.relplot(data=df, x="Age", y="MonthlyIncome", row="Gender", col="Department",
                hue="Attrition", height=3.2, alpha=0.7, palette="Set1")
g.figure.suptitle("7. facet_grid: Gender (rows) x Department (cols)", y=1.02)
plt.show()
print("7. Observation: income rises with age in every panel; attrition (Yes) points cluster "
      "at lower incomes and younger ages, and the pattern is similar for both genders, "
      "with Sales showing the widest income spread.\n")

# 8. Pairs plot (4 numerical variables)
pair_vars = ["Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany"]
g = sns.pairplot(df, vars=pair_vars, hue="Attrition", corner=False, plot_kws={"alpha": 0.5},
                 palette="Set1")
g.figure.suptitle("8. Pairs plot", y=1.02)
plt.show()
print("8. Correlation matrix of selected variables\n", df[pair_vars].corr().round(2))
print("   Observation: MonthlyIncome, TotalWorkingYears, Age and YearsAtCompany are all "
      "positively correlated (income vs working years is the strongest, r ~ 0.77).\n")

# 9. Parallel coordinate plot (min-max scaled so all axes are comparable)
pc_vars = ["Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany", "DistanceFromHome"]
pc = df[pc_vars + ["Attrition"]].sample(300, random_state=2).copy()
pc[pc_vars] = (pc[pc_vars] - pc[pc_vars].min()) / (pc[pc_vars].max() - pc[pc_vars].min())
plt.figure(figsize=(10, 6))
parallel_coordinates(pc, "Attrition", color=["#1f77b4", "#d62728"], alpha=0.4)
plt.title("9. Parallel Coordinate Plot (scaled 0-1), grouped by Attrition")
plt.ylabel("Scaled value")
plt.tight_layout(); plt.show()
