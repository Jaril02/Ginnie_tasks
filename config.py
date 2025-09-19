from pathlib import Path


BASE_DIR= Path(__file__).resolve().parent
INPUT_DIR=  BASE_DIR/"input"
OUTPUT_DIR= BASE_DIR/"output"
CLEAN_DIR=BASE_DIR/"clean"

for folder in [OUTPUT_DIR,CLEAN_DIR]:
    folder.mkdir(parents=True,exist_ok=True)