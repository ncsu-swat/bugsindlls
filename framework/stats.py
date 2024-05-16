import csv
import sys
import os 

## Trie (Implementation taken from https://stackoverflow.com/questions/67769308/how-to-print-trie-in-tree-structure-in-python)

class TrieNode:
    def __init__(self):
        self.is_end = False
        self.children = {}
        self.visited = 1

class Trie(object):
    def __init__(self):
        self.root = TrieNode()

    def insert(self, array):
        node = self.root
        node.visited += 1
 
        for x in array:
            if x in node.children:
                node = node.children[x]
                node.visited += 1
            else:
                child = TrieNode()
                node.children[x] = child
                node = child            
        node.is_end = True

    def __repr__(self):
        def recur(node, indent):
            return "".join(indent + "|-- " + key + " (" + str(child.visited) + ") "
                                  + recur(child, indent + "  ") 
                for key, child in node.children.items())

        return recur(self.root, "\n")
    
#####################################

def process_file(libname, print_fmt):
    tr = Trie()
    print('----------------------------')
    print(libname)
    print('----------------------------')
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
                if (print_fmt == "rows") or (print_fmt == "both"):
                    print(f'\t{row["Issue #"]} {row["Device"]} {row["Buggy File(s)"]}')

                if row["Buggy File(s)"] > "":
                    for file in row["Buggy File(s)"].split(","):
                        file = file.strip()
                        tr.insert(file.split("/"))

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

        print(f'  # Require GPU: {num_gpus} ({num_gpus*100/(num_gpus+num_cpus):.2f}%)')
        print(f'  # Do NOT Require GPU: {num_cpus} ({num_cpus*100/(num_gpus+num_cpus):.2f}%)')        
        print(f'  # C: {num_c} ({num_c*100/(num_c+num_py):.2f}%)')
        print(f'  # Python: {num_py} ({num_py*100/(num_c+num_py):.2f}%)')
    
    if (print_fmt == "trie") or (print_fmt == "both"):
        print('----------------------------')
        print(tr)             
        print('\n----------------------------\n')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('You need to pass the name of the library. For example, $> python stats.py jax')
        sys.exit(1)
    
    if len(sys.argv) < 3:
        # rows: Print rows
        # trie: Print buggy files in a trie
        # both: Print the rows AND the trie
        # none: Do not print lines
        print("Printing in trie format by default")
        sys.argv.append("trie")
    
    process_file(sys.argv[1], sys.argv[2])
