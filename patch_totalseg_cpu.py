import fileinput

for line in fileinput.input("auto_train.py", inplace=True):
    if "TotalSegmentator" in line and "cmd" in line:
        print(line.rstrip() + ", '--device', 'cpu']")
    else:
        print(line, end="")
