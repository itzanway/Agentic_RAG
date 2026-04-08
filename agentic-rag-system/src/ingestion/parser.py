from unstructured.partition.pdf import partition_pdf

def parse_document(file_path):
    print(f"Parsing document: {file_path}...")
    # This extracts elements while attempting to preserve table structures
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res", # Uses advanced layout detection
        infer_table_structure=True
    )
    
    # Separate tables from standard text chunks
    texts = [str(el) for el in elements if el.category != "Table"]
    tables = [str(el) for el in elements if el.category == "Table"]
    
    print(f"Extracted {len(texts)} text blocks and {len(tables)} tables.")
    return texts, tables

if __name__ == "__main__":
    # Test it by pointing it to a sample PDF in your data/raw_docs folder
    # parse_document("../../data/raw_docs/sample.pdf")
    pass