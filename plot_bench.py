import matplotlib.pyplot as plt
import pandas as pd
import sys

csv_file_name = sys.argv[1]

df = pd.read_csv(csv_file_name)
# print(df.columns)
df = df[['Benchmark', 'k', 'GPU Time (sec)', 'Noise.1']]
# print(df)
grouped = df.groupby('Benchmark')
# print(grouped.groups)

fig = plt.figure()

for name, group in grouped:
    # print(name)
    # print(group)
    x = group['k']
    y = group['GPU Time (sec)']
    yerr = group['Noise.1'] * y

    plt.errorbar(x, y, yerr=yerr, label=name)

plt.legend()
# plt.show()
plt.savefig("./bench_result.png", dpi=300)
