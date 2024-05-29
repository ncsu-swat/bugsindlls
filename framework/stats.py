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
        
        ##
        bug_manifest_type = {}
        ##
        num_cpus = 0
        num_gpus = 0
        ##
        num_c = 0
        num_py = 0      
        num_cuda = 0  
        for row in csv_reader:
            if row["Reproduced"] == "Yes":
                if (print_fmt == "rows") or (print_fmt == "both"):
                    print(f'\t{row["Issue #"]} {row["Device"]} {row["Buggy File(s)"]}')
                
                ## bug types
                bug_m_type = row["Type"].strip().lower()
                if bug_m_type not in bug_manifest_type:
                    bug_manifest_type[bug_m_type] = 1
                else:
                    bug_manifest_type[bug_m_type] += 1

                ## cpu/gpu
                device = row["Device"].strip()
                if device == "CPU":
                    num_cpus += 1
                elif device == "GPU":
                    num_gpus += 1
                
                if row["Buggy File(s)"] > "":
                    for file in row["Buggy File(s)"].split(","):
                        file = file.strip()
                        tr.insert(file.split("/"))

                        # c/cuda native/python
                        if ".cu" in file:
                            num_cuda += 1
                        elif ".c" in file or ".cc" in file or ".h" in file or ".mm" in file:
                            num_c += 1
                        elif ".py" in file or ".pyi" in file:
                            num_py += 1                        

        print(f'  # Require GPU: {num_gpus} ({num_gpus*100/(num_gpus+num_cpus):.2f}%)')
        print(f'  # Do NOT Require GPU: {num_cpus} ({num_cpus*100/(num_gpus+num_cpus):.2f}%)')        
        print(f'  # C/CPP: {num_c} ({num_c*100/(num_c+num_py+num_cuda):.2f}%)')
        print(f'  # CUDA Native: {num_cuda} ({num_cuda*100/(num_c+num_py+num_cuda):.2f}%)')
        print(f'  # Python: {num_py} ({num_py*100/(num_c+num_py+num_cuda):.2f}%)')
        print('----------------------------')
        print(f'  # Bug manifestation types:')
        bug_manifest_type = dict(sorted(bug_manifest_type.items(), key=lambda item: item[1], reverse=True)) # sort by number of occurance
        for bug_m_type in bug_manifest_type.keys():
            print(f'    {bug_m_type}: {bug_manifest_type[bug_m_type]}')
    
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
    
    if sys.argv[1] == 'all':
        for libname in ['jax', 'pytorch', 'tensorflow']:
            process_file(libname, sys.argv[2])
    else:
        process_file(sys.argv[1], sys.argv[2])
