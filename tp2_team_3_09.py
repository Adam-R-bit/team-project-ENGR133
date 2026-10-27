"""
Course Number: ENGR 13300
Semester: Fall 2026
Description:
    Replace this line with a description of your program.

Assignment Information:
    Assignment:     tp2 team 3
    Team ID:        LC9
    Author:         Guen Batsaihan, gbatsaih@purdue.edu
    Date:           e.g. 01/23/2026

Contributors:
    Name, login@purdue [repeat for each]

    My contributor(s) helped me:
    [ ] understand the assignment expectations without
        telling me how they will approach it.
    [ ] understand different ways to think about a solution
        without helping me plan my solution.
    [ ] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.

Academic Integrity Statement:
    I have not used source code obtained from any unauthorized
    source, either modified or unmodified; nor have I provided
    another student access to my code.  The project I am
    submitting is my own original work.
"""

import matplotlib.pyplot as plt
import pandas as pd


def scale_features(df, feature_cols):
    for col in feature_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        df[col] = (df[col]- mean_val) / std_val


def main():
    csv_path = input("Enter the name of CSV image features file: "\n)
    scaling_choice = input("Do you want to apply feature scaling? (yes/no): "\n).strip()
    df = pd.read_csv(csv_path)

    feature_cols = ["hue_mean", "hue_std", "saturation_mean", "saturation_std","value_mean", "value_std","number_lines","has_circle","has_triangle"]

    if scaling_choice.lower() == "yes":
        print("Applying feature scaling to the dataset...")
        scale_features(df, feature_cols)
        print("Feature scaling applied successfully.")
    else:
        print("Processing without feature scaling.")

    print(f"Feature Dataset Shape: {df.shape}")
    print("Feature Dataset Head:")
    print(df.head())

    x_feat = input("Enter the column to use as the x-axis feature: "\n).strip()
    y_feat = input("Enter the column to use as the y-axis feature: "\n).strip()
    z_feat = input("Enter the column to use as the z-axis feature (or press Enter to skip): ").strip()

    fig = plt.figure(figsize=(8, 6))

    if z_feat:
        ax = fig.add_subplot(projection="3d")
    else:
        ax = fig.add_subplot()

    group0 = df[df["ClassId"] == 0]
    group1 = df[df["ClassId"] == 1]

    if z_feat:
        ax.scatter(group0[x_feat], group0[y_feat], group0[z_feat], c="red",alpha=0.6, label="Deer Warning (0)")
        ax.scatter(group1[x_feat], group1[y_feat], group1[z_feat], c="blue", alpha=0.6, label="Left Turn Ahead (1)",)
        ax.set_zlabel(z_feat)
    else: 
        ax.scatter(group0[x_feat], group0[y_feat], c="red", alpha=0.6, label="Deer Warning (0)")
        ax.scatter(group1[x_feat], group1[y_feat], c="blue", alpha=0.6, label="Left Turn Ahead (1)",)

    ax.set_xlabel(x_feat)
    ax.set_ylabel(y_feat)
    ax.legend()

    plt.savefig("KNN_feature_space.png")
    print("Scatter plot saved to KNN_feature_space.png")

if __name__ == "__main__":
    main()
