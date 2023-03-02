import os, glob, re

def lines_that_start_with(string, fp):
    for line in fp:
        if line.startswith(string):
            return line.split()[2]
    return "N/A"

for filename in glob.glob('./optimized_small_*.out'):
   with open(os.path.join(os.getcwd(), filename), 'r') as f:
       overall_accuracy = lines_that_start_with("overall accuracy", f)
       print(f"{filename.replace('./for_loops/optimize_', '').replace('.out', '')},{overall_accuracy}")
