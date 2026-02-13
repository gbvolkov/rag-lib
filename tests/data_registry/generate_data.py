import pandas as pd
import os

# Base paths
BASE_CSV = "tests/data_registry/csv"
BASE_PDF = "tests/data_registry/pdf"
BASE_EXCEL = "tests/data_registry/excel"
BASE_TEXT = "tests/data_registry/text"

os.makedirs(BASE_TEXT, exist_ok=True)

# 1. CSV Generation
def generate_csvs():
    # Standard
    df = pd.DataFrame({'id': range(50), 'name': [f'Item {i}' for i in range(50)], 'value': range(100, 150)})
    df.to_csv(f"{BASE_CSV}/standard.csv", index=False)
    
    # Pipe Delimited
    df.to_csv(f"{BASE_CSV}/pipes.csv", sep='|', index=False)
    
    # Latin-1 Encoded
    with open(f"{BASE_CSV}/latin1.csv", 'w', encoding='latin-1') as f:
        f.write("id,name,cost\n1,Café,10.5\n2,El Niño,20.0")

    # Empty
    with open(f"{BASE_CSV}/empty.csv", 'w') as f:
        pass

    # No Header (Just data)
    with open(f"{BASE_CSV}/no_header.csv", 'w') as f:
        f.write("1,Apple,Red\n2,Banana,Yellow")

# 2. Excel Generation
def generate_excels():
    # Multi-sheet
    with pd.ExcelWriter(f"{BASE_EXCEL}/multisheet.xlsx") as writer:
        pd.DataFrame({'Sheet': ['One']}).to_excel(writer, sheet_name='Sheet1')
        pd.DataFrame({'Sheet': ['Two']}).to_excel(writer, sheet_name='Sheet2')

# 3. PDF Generation (Dummy/Corrupt)
def generate_pdfs():
    # Corrupt PDF (Random bytes)
    with open(f"{BASE_PDF}/corrupt.pdf", 'wb') as f:
        f.write(b'%PDF-1.4\n...garbage...')
    
    # Valid-ish empty text file renamed to PDF (to test non-binary failure if any)
    with open(f"{BASE_PDF}/not_a_pdf.pdf", 'w') as f:
        f.write("This is just text.")

# 4. Text Generation
def generate_text():
    # Semantic Text (Repeated sentences for similarity)
    text = "The quick brown fox jumps over the lazy dog. " * 20
    text += "\n\n"
    text += "Machine learning is fascinating. " * 20
    with open(f"{BASE_TEXT}/semantic.txt", 'w') as f:
        f.write(text)

if __name__ == "__main__":
    print("Generating Test Data Registry...")
    generate_csvs()
    generate_excels()
    generate_pdfs()
    generate_text()
    print("Done.")
