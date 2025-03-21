import sys
import os
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <library_name>")
        return
    library_name = sys.argv[1]
    apis_file = f"data/{library_name}_APIdef.txt"
    supported_apis = []

    with open(apis_file, 'r') as file:
        apis = file.read()

        for api in apis.splitlines():
            supported_apis.append(api.split("(")[0].strip())

    with open("/tmp/apis_under_test.txt", 'r') as file:
        apis_text = file.read()
        print(f"\nAPIs under test:\n{apis_text}\n")
        apis_under_test = apis_text.splitlines()

    apis_to_skip = []
    
    for api in supported_apis:
        if api not in apis_under_test:
            apis_to_skip.append(api)        

    skip_filename = f"src/config/skip_{library_name}.txt"
    with open(skip_filename, 'w') as file:
        for api in apis_to_skip:
            file.write(api + '\n')

if __name__ == "__main__":
    main()