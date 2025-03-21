import sys
import os
import subprocess
from rapidfuzz import fuzz

def main():
    if len(sys.argv) < 2:
        print("Usage: python match_bugs.py <config_name>")
        return
    config_name = sys.argv[1]

    config_file = f"src/config/{config_name}"

    library_name = None
    output = None

    with open(config_file, 'r') as file:
        config_data = file.read()
        for line in config_data.splitlines():
            if line.startswith("libs"):
                library_name = line.split('=')[1].strip()
            elif library_name is not None and line.startswith(f"{library_name}_output"):
                output = line.split('=')[1].strip()
                break

    output_folder = f"src/{output}"

    with open("/tmp/apis_under_test.txt", 'r') as file:
        apis_under_test = file.read().splitlines()

    output_msgs = {}
    for api in apis_under_test:
        output_msgs[api] = ""

    oracles = [name for name in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, name))]
    
    print('\n')
    
    for oracle in oracles:
        categories = [name for name in os.listdir(f"{output_folder}/{oracle}") if os.path.isdir(os.path.join(f"{output_folder}/{oracle}", name))]
        for category in categories:
            if category.endswith("bug"):
                apis_under_category = [name for name in os.listdir(f"{output_folder}/{oracle}/{category}") if os.path.isdir(os.path.join(f"{output_folder}/{oracle}/{category}", name))]
                if len(apis_under_category) == 0:
                    print(f"No violation of {oracle} in the {category} category")
                for api in apis_under_category:
                    python_scripts = os.listdir(f"{output_folder}/{oracle}/{category}/{api}")
                    if len(python_scripts) == 0:
                        print(f"{api}: No violation of {oracle} in the {category} category")
                    for script in python_scripts:
                        # execute each script, store the output
                        script_path = os.path.join(output_folder, oracle, category, api, script)
                        result = subprocess.run(['python', script_path], capture_output=True, text=True)
                        # print(f"Output of {script_path}:\n{result.stdout}")
                        # if result.stderr:
                        #     print(f"Error in {script_path}:\n{result.stderr}")
                        output_msgs[api] += result.stdout + '\n'
                        if result.stderr:
                            output_msgs[api] += result.stderr + '\n'
    
    print('\n')
    
    with open("/tmp/error_list.txt", "r") as file:
        reference_error = file.read()
        n_bugs = len(reference_error.splitlines())
        reproduced = 0
    
    for api in output_msgs:
        if output_msgs[api] > "":
            print(f"-> Output for {api}:\n{output_msgs[api]}")
            best_score = 0
            best_matches = (None, None)
            for msg in output_msgs[api].splitlines():
                if len(msg.strip()) < 2:
                    continue
                for ref_line in reference_error.splitlines():
                    if len(ref_line.strip()) < 2:
                        continue
                    similarity = fuzz.partial_ratio(msg.lower(), ref_line.lower())
                    if similarity > best_score:
                        best_score = similarity
                        best_matches = (msg, ref_line)
            
            if best_matches[0] is not None:
                print(f"Best matching pair of lines: {best_matches}")
            
            if best_score > 70:
                print(f"Fuzzy match found with a score of {best_score}, the bug was most likely reproduced")
                reproduced += 1
                
            else:
                print(f"Only a score of {best_score}, the bug was most likely not reproduced")
        
        else:
            print(f"-> {api} did not face any failures")

    print(f"\nReproduced {reproduced} out of {n_bugs} bugs")
if __name__ == "__main__":
    main()