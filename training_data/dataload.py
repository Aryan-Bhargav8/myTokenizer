from datasets import load_dataset
import os



def download_openwebtext(target_size_gb=2, output_path="./training_data/raw/openwebtext.txt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    target_bytes = int(target_size_gb * 1024 * 1024 * 1024)
    total_bytes = 0
    num_docs = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for example in ds:
            text = example["text"]

            if len(text) < 100:
                continue

            # AGGRESSIVE CLEANING - remove ALL control characters
            # Keep only: printable ASCII (32-126), common unicode, newlines, tabs
            cleaned = []
            for c in text:
                code = ord(c)
                # Keep printable ASCII
                if 32 <= code <= 126:
                    cleaned.append(c)
                # Keep newline, tab, carriage return
                elif c in '\n\t\r':
                    cleaned.append(c)
                # Keep unicode letters and common punctuation (above 127)
                elif code >= 127:
                    cleaned.append(c)
                # Everything else (control chars 0-31 except \n\t\r) gets dropped
            
            text = ''.join(cleaned)
            
            if not text.strip():
                continue

            f.write(text) #writing in the file
            # f.write("\n")  # document separator, cuz we are writing in one single file txt and the dataset has many docs kinda

            total_bytes += len(text.encode("utf-8"))
            num_docs += 1

            if num_docs % 5000 == 0:
                print(f" {num_docs} docs : {total_bytes / 1e9:.2f} GB")

            if total_bytes >= target_bytes:
                break
    
    print(f"\nDone. {num_docs} docs downloaded")
    print(f"Saved to: {output_path}")



if __name__ == "__main__":
    download_openwebtext(target_size_gb=1)