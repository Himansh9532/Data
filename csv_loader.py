from langchain_community.document_loaders.csv_loader import CSVLoader

# Initialize the CSVLoader with the path to the CSV file
loader = CSVLoader(file_path="stud.csv")

# Load the data
data = loader.load()

# Optional: print or inspect the data
print(data)
