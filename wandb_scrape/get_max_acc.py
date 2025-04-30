import wandb
import pandas as pd

# wandb.login()

api = wandb.Api()

entity = 'slk2191-columbia-university'
project = 'learned-tree-tests'


# Filtering criteria
n = 10  # number of recent runs to keep
target_dataset = 'flowers102'
target_without_classical = 1
target_num_ways = 5
target_num_shots = 1
target_tree = \
'''root (AugmentationType.DEPTH L_prob: 0.571, R_prob: 0.429)
  L (AugmentationType.NERF)
  R (AugmentationType.DEPTH)'''

# Descriptive name for classical
classical_str = "without_classical" if target_without_classical else "with_classical"

# Fetch all runs
runs = api.runs(f"{entity}/{project}")

# Filter runs based on config
filtered_runs = []
for run in runs:
    config = run.config
    summary = run.summary
    if (
        config.get("dataset") == target_dataset and
        config.get("without_classical") == target_without_classical and
        config.get("num_shots") == target_num_shots and
        config.get("num_ways") == target_num_ways and
        summary.get("tree") == target_tree
    ):
        filtered_runs.append(run)

# Take the most recent n runs
filtered_runs = sorted(filtered_runs, key=lambda r: r.created_at, reverse=True)[:n]

# Collect accuracy data
data = []
for run in filtered_runs:
    try:
        history = run.history(keys=["test_accuracy"])
        if "test_accuracy" in history:
            max_acc = history["test_accuracy"].max()
            # get last non-zero final accuracy, since sometimes we only log every 5 epochs
            test_acc_series = history["test_accuracy"]
            final_acc = 0.0
            for acc in reversed(test_acc_series):
                if acc > 0:
                    final_acc = acc
                    break
        else:
            max_acc = None
            final_acc = None

        data.append({
            "run_name": run.name,
            "final_accuracy": final_acc,
            "max_accuracy": max_acc
        })

        print(run.name, final_acc, max_acc)

    except Exception as e:
        print(f"Error processing run {run.name}: {e}")

df = pd.DataFrame(data)

filename = f"{target_dataset}_{classical_str}_{target_num_ways}_num_ways_{target_num_shots}_shots.csv"
df.to_csv(filename, index=False)
print(f"Saved filtered results to {filename}")

# for checking computation is right with what we prev calculated
avg_final = df["final_accuracy"].mean() * 100
std_final = df["final_accuracy"].std() * 100
avg_max = df["max_accuracy"].mean() * 100
std_max = df["max_accuracy"].std() * 100

print(f"final acc: {avg_final:.2f} +- {std_final:.2f}")
print(f"max acc: {avg_max:.2f} +- {std_max:.2f}")

# Print running averages
print("\nRunning averages (for debugging):")
final_accs = df["final_accuracy"].tolist()
max_accs = df["max_accuracy"].tolist()
# reverse here since we reverse above - we want the lists in order from first run completed to last run completed
# since the averages we put in the spreadsheet might just be from the first 3 runs and by now 10 have completed
# doing this for a sanity check so that we can see that the average value calculations match with what we put in 
# the spreadsheet
final_accs.reverse()
max_accs.reverse()

for i in range(1, len(final_accs) + 1):
    running_final_avg = pd.Series(final_accs[:i]).mean() * 100
    running_final_std = pd.Series(final_accs[:i]).std() * 100
    running_max_avg = pd.Series(max_accs[:i]).mean() * 100
    running_max_std = pd.Series(max_accs[:i]).std() * 100

    print(f"[{i}] Final: {running_final_avg:.2f}% +- {running_final_std:.2f}%, Max: {running_max_avg:.2f}% +- {running_max_std:.2f}%")
