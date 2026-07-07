# Usage: poetry run python src/ingestion/splitting.py [file_path]
import os
import pymupdf4llm
import sys
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import argparse
from qdrant_client import QdrantClient, models
import uuid


def parse_and_split(file_path, output_file_path, split_only=False, pages=None):
    print(f"Parsing and splitting file: {file_path} with split_only={split_only}")
    if (not split_only):
        start, end = None if pages is None else map(int, pages.split('-'))
        pages = None if pages is None else range(start, end)
        extract = pymupdf4llm.to_markdown(
            file_path, header=False, footer=False, show_progress=True, force_ocr=True, pages=pages)
        # Clean bullet points
        extract = re.sub(r'^\s*- ;\s*', '- ', extract, flags=re.MULTILINE)
        extract = re.sub(r'^\s*;\s*', '- ', extract, flags=re.MULTILINE)
        extract = re.sub(r'^\s*;\s*$', '', extract, flags=re.MULTILINE)
        extract = re.sub(r'^\s*\*\*\d+\*\*\s*$', '', extract, flags=re.MULTILINE)
        # Save the markdown content to a file
        open(output_file_path, 'w').write(extract)

    headers_to_split_on = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
        ("####", "H4"),
        ("#####", "H5"),
        ("######", "H6")
    ]

    md = open(output_file_path, 'r').read()

    # MD splits
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(md)

    # Char-level splits
    chunk_size = 250
    chunk_overlap = 30
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Split are tuples of (metadata, page_content)
    return text_splitter.split_documents(md_header_splits)


def main():
    output_path = "data/processed"

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path")
    parser.add_argument("-s", "--split", action="store_true")
    parser.add_argument("-p", "--pages", type=str, default=None,
                        help="Indicate which pages to process by providing page start and end indices separated by a hyphen. If not provided, all pages will be processed.")
    args = parser.parse_args()

    file_path = args.file_path
    file_name = file_path.split("/")[-1]
    split_only = args.split
    pages = args.pages
    print(f"Processing file: {file_path} with split_only={split_only} and pages={pages if pages is not None else 'all'}")
    output_file_path = output_path + "/" + \
        file_name.replace(".pdf", ".md")

    if (split_only and not os.path.exists(output_file_path)):
        raise FileNotFoundError(
            f"File {output_file_path} does not exist. Please run without --split to convert the file first.")

    splits = parse_and_split(file_path, output_file_path, split_only=split_only, pages=pages)

    qdrant_client = QdrantClient(path="db")

    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # Good compromise memory accuracy
    collection_name = "shift_project_agriculture"

    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name,
            vectors_config=models.VectorParams(
                size=qdrant_client.get_embedding_size(model_name),
                distance=models.Distance.COSINE
            )
        )
        print("Collection créée avec succès.")
    else:
        print("La collection existe déjà.")

    metadata_with_docs = [
        {"document": split.page_content, "source": file_name, "section_title": split.metadata} for split in splits
    ]

    qdrant_client.upload_collection(
        collection_name,
        vectors=[models.Document(text=split.page_content, model=model_name) for split in splits],
        payload=metadata_with_docs,
        ids=[uuid.uuid4() for _ in range(len(splits))]
    )

    print(f"Done processing {file_path}. Total chunks sent: {len(splits)}")

    sys.exit(0)


if __name__ == "__main__":
    main()
