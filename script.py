import pandas as pd

def meanvalue(data):
    return sum(data) / len(data)

def meandeviation(data):
    mu = meanvalue(data)
    variance = sum((x - mu) ** 2 for x in data) / len(data)
    return variance ** 0.5

def mean_absolute_error(actual, predicted):
    n = len(actual)
    total_error = 0

    for i in range(n):
        error = abs(actual[i] - predicted[i])
        total_error += error

    return total_error / n

def correlation(dataset1, dataset2):
    n = len(dataset1)
    mu1 = sum(dataset1) / n
    mu2 = sum(dataset2) / n
    numerator = 0
    sum_sq_diff1 = 0
    sum_sq_diff2 = 0
    for i in range(n):
        diff1 = dataset1[i] - mu1
        diff2 = dataset2[i] - mu2
        numerator += diff1 * diff2
        sum_sq_diff1 += diff1 ** 2
        sum_sq_diff2 += diff2 ** 2
    denominator = (sum_sq_diff1 * sum_sq_diff2) ** 0.5
    return numerator / denominator if denominator != 0 else 0

marks = pd.read_csv('Teen_Mental_Health_Dataset.csv')

columns = marks.columns.tolist()
print("Columns in dataset:", columns)

dataset1 = marks['anxiety_level']
dataset2 = marks['sleep_hours']

print(f"Mean Deviation (Anxiety): {meandeviation(dataset1):.2f}")
print(f"Correlation (Anxiety vs Sleep): {correlation(dataset1, dataset2):.4f}")

numeric_marks = marks.select_dtypes(include=['number'])
cols = numeric_marks.columns
for i in cols:
    for j in cols:
        if i < j:
            result = correlation(marks[i], marks[j])
            print(f"{i} vs {j}: {result:.4f}")
print("-"*100)
for i in cols:
    actual_values = numeric_marks[i]
    avg_val = meanvalue(actual_values)
    dev_val = meandeviation(actual_values)

    baseline_predictions = [avg_val] * len(actual_values)
    mae_val = mean_absolute_error(actual_values, baseline_predictions)

    print(f"Column: {i.upper()}")
    print(f"  - Mean Value:         {avg_val:.4f}")
    print(f"  - Mean Deviation:     {dev_val:.4f}")
    print(f"  - Baseline MAE:       {mae_val:.4f}")
    print("-" * 50)