import sys, os, csv

def main():
    if len(sys.argv) < 3:
        print("Usage: python retrieve.py <colname> <library_name> <version>")
        return
    
    colname = sys.argv[1]
    libname = sys.argv[2]
    version = sys.argv[3]
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(f'{dir_path}/../bug_dataset_{libname}.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(filter(lambda row: not row.startswith(",Filter"), csv_file))
        for row in csv_reader:
            if row["Reproduced"] == "Yes":
                if row["Buggy Version"].strip() == version.strip():
                    print(row[colname])
    

if __name__ == "__main__":
    main()