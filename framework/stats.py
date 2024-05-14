import csv
import sys
import os 


def process_file(libname):
    print(libname)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(f'{dir_path}/../bug_dataset_{libname}.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(filter(lambda row: not row.startswith(",Filter"), csv_file))
        line_count = 0
        first = True
        ##
        num_cpus = 0
        num_gpus = 0
        ##
        num_c = 0
        num_py = 0        
        for row in csv_reader:
            if first:
                first = False
                continue
            if row["Reproduced"] == "Yes":
                print(f'\t{row["Issue #"]} {row["Device"]} {row["Buggy File(s)"]}')

                ## cpu/gpu
                if row["Device"] == "CPU":
                    num_cpus += 1
                elif row["Device"] == "GPU":
                    num_gpus += 1

                # c/python
                if ".c" in row["Buggy File(s)"] or ".cc" in row["Buggy File(s)"] or ".h" in row["Buggy File(s)"]:
                    num_c += 1
                elif ".py" in row["Buggy File(s)"]:
                    num_py += 1

        print(f'  # Require GPU: {num_gpus} ({num_gpus/(num_gpus+num_cpus):.2f}%)')
        print(f'  # Require CPU: {num_cpus} ({num_cpus/(num_gpus+num_cpus):.2f}%)')        
        print(f'  # C: {num_c} ({num_c/(num_c+num_py):.2f}%)')
        print(f'  # Python: {num_py} ({num_py/(num_c+num_py):.2f}%)')                

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('You need to pass the name of the library. For example, $> python stats.py jax')
        sys.exit(1)
    
    process_file(sys.argv[1])
