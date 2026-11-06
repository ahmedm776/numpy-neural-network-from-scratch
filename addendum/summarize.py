import json, glob, statistics as st
rows = []
print("| Arm | Seeds | Val acc | Test acc (official 10k) | test[0:5700] | test[5700:10000] |\n|---|---|---|---|---|---|")
for a, label in (("scratch", "NumPy scratch (SGD 0.5, bs128, 100 ep)"), ("keras_sgd", "Keras, matched (SGD 0.5, bs128, 100 ep, N(0,0.01) init)"), ("keras_adam", "Keras as shipped (Adam, bs32, 10 ep)")):
    r = [json.load(open(f)) for f in sorted(glob.glob(f"results/{a}_seed*.json"))]
    f = lambda k: f"{100*st.mean(x[k] for x in r):.2f} ± {100*st.stdev(x[k] for x in r):.2f}"
    print(f"| {label} | {len(r)} | {f('val_acc')} | {f('test_acc')} | {f('test_0_5700')} | {f('test_5700_10000')} |")
