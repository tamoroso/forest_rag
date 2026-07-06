# Usage: poetry run python src/ingestion/splitting.py [file_path]
import os
import pymupdf4llm
import sys
import re
import chromadb
import time
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import tiktoken
import argparse
import config


# TODO: Add to utils.py


def count_tokens(text):
    # Use tiktoken to estimate embbeding costs
    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    return len(enc.encode(text))


def add_to_collection(file_path, collection, chunk_to_send):
    try:
        print(f"Adding {len(chunk_to_send)} chunks to the collection...")
        collection.add(
            ids=[
                f"{file_path.split('/')[-1]}_{j}" for j in range(len(chunk_to_send))],
            documents=[s.page_content for s in chunk_to_send],
            metadatas=[s.metadata for s in chunk_to_send]
        )
    except Exception as e:
        print(f"Error occurred while adding to collection: {e}")


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


def count_tokens_in_splits(splits):
    tokens = []
    num_tokens = 0

    for split in splits:
        # Estimate the number of tokens in the content
        split_tokens = count_tokens(split.page_content)
        num_tokens += split_tokens
        tokens.append(split_tokens)

    print(f'Total number of tokens: {num_tokens}')

    return tokens


def send_chunks_to_collection(file_path, collection, splits):
    # Send the chunks to the collection
    print(f"Sending {len(splits)} chunks to the collection...")
    chunk_size_limit = 40000
    chunk_token_count = 0
    chunk_to_send = []
    window_start = time.time()
    tokens = count_tokens_in_splits(splits)

    for i, split in enumerate(splits):
        print(f"Processing chunk {i+1}/{len(splits)}")
        if tokens[i] + chunk_token_count < chunk_size_limit:
            chunk_token_count += tokens[i]
            print(f"Adding chunk {i+1} to the current chunk. Current token count: {chunk_token_count}")
            chunk_to_send.append(split)
        else:
            print(f"Chunk {i+1} exceeds the token limit. Sending current chunk to collection. Current token count: {chunk_token_count}")
            elapsed = time.time() - window_start
            if elapsed < 60:
                time.sleep(60 - elapsed)
            window_start = time.time()
            # Send the current chunk to the collection
            print("Adding chunk to collection...")
            add_to_collection(file_path, collection, chunk_to_send)
            # Reset the chunk
            chunk_token_count = 0
            chunk_to_send = [split]
    # Purge the last chunk if it's not empty
    if len(chunk_to_send) > 0:
        print(f"Sending final chunk to collection. Current token count: {chunk_token_count}")
        print("Adding chunk to collection...")
        add_to_collection(file_path, collection, chunk_to_send)


def main():
    output_path = "data/processed"

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path")
    parser.add_argument("-s", "--split", action="store_true")
    parser.add_argument("-p", "--pages", type=str, default=None,
                        help="Indicate which pages to process by providing page start and end indices separated by a hyphen. If not provided, all pages will be processed.")
    args = parser.parse_args()

    file_path = args.file_path
    split_only = args.split
    pages = args.pages
    print(f"Processing file: {file_path} with split_only={split_only} and pages={pages if pages is not None else 'all'}")
    output_file_path = output_path + "/" + \
        file_path.split("/")[-1].replace(".pdf", ".md")

    if (split_only and not os.path.exists(output_file_path)):
        raise FileNotFoundError(
            f"File {output_file_path} does not exist. Please run without --split to convert the file first.")

    splits = parse_and_split(file_path, output_file_path, split_only=split_only, pages=pages)

    chromadb_client = chromadb.PersistentClient("db")

    collection = chromadb_client.get_or_create_collection(
        name="forest_rag",
        embedding_function=OpenAIEmbeddingFunction(
            api_key=config.OPENAI_API_KEY,
            model_name="text-embedding-3-small"
        ))

    send_chunks_to_collection(file_path, collection, splits)

    print(f"Done processing {file_path}. Total chunks sent: {len(splits)}")

    sys.exit(0)


if __name__ == "__main__":
    main()
