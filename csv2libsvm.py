import csv

def csv_to_libsvm(input_csv, output_libsvm):
    with open(input_csv, 'r') as csv_file, open(output_libsvm, 'w') as libsvm_file:
        csv_reader = csv.reader(csv_file)
        
        # Skip the header row
        header = next(csv_reader)
        
        for row in csv_reader:
            label = row[0]  # Assuming the first column contains labels
            features = row[1:]  # Assuming the remaining columns are features
            libsvm_line = label + " " + " ".join([f"{i+1}:{val}" for i, val in enumerate(features) if val != '0'])
            libsvm_file.write(libsvm_line + '\n')

# Example usage:
csv_to_libsvm("bow.train.csv", "bow.train.libsvm")
csv_to_libsvm("bow.test.csv", "bow.test.libsvm")
csv_to_libsvm("bow.eval.anon.csv", "bow.eval.anon.libsvm")
