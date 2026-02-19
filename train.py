import os
import json
from collections import Counter
from pretokenize import pretokenize



# Special Tokens ------------------------------
SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3
}


# Helpers --------------------------------------

def read_corpus(path: str, max_lines: int = None) -> str:
    """Read Corpus from file, while limiting to max_lines optionally"""

    with open(path, "r", encoding="utf-8") as f:
        if max_lines:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line)
            return "".join(lines)
        return f.read()
    

# this is to convert the pretokenized string chunks into corresponding bytes
def chunk_to_bytes(chunk: str) -> tuple[int]:
    """Convert a string chunk into its corresponding bytes, then convert it to a tuple of byte integers.
    eg: 'hello' -> (some integers)
    And we are using tuples cuz they could be used as dictionary keys"""

    return tuple(chunk.encode("utf-8"))


# to have a base vocabulary before expanding it using BPE
def build_base_vocab() -> dict:
    """
    Build base vocabulary with:
    - 4 special tokens
    - 256 byte tokens
    """

    vocab = dict(SPECIAL_TOKENS) #starts with special tokens

    #Now we add the rest 256 possible bytes
    for byte_val in range(256):
        byte_token = bytes([byte_val]).decode("latin-1", errors="replace") #this first wraps the integers,then converts it to bytes and then decodes that byte to a string for the vocabulary, basically single byte as a string

        vocab[byte_token] = len(vocab) #next available id

    return vocab


# Pair Counting ------------------------------------------------

def count_pairs(corpus_chunks: list[tuple[int]]) -> Counter:
    """
    Counts frequency of all adjacent characters across the corpus of text.
    Each chunk is a tuple of byte integers
    """

    pair_counts = Counter()

    for chunk in corpus_chunks:
        for i in range(len(chunk) - 1):
            pair = (chunk[i], chunk[i+1])
            pair_counts[pair] += 1

    return pair_counts


## Taken help from ai for parallelizing the pair counting
from multiprocessing import Pool
from functools import partial

def count_pairs_parallel(corpus_chunks: list[tuple[int]], num_workers: int = 4) -> Counter:
    """
    Split corpus into chunks, count pairs in parallel, then merge counts.
    """
    if len(corpus_chunks) < 1000:  # not worth parallelizing small corpus
        return count_pairs(corpus_chunks)
    
    # Split corpus into chunks for each worker
    chunk_size = len(corpus_chunks) // num_workers
    splits = [
        corpus_chunks[i:i + chunk_size] 
        for i in range(0, len(corpus_chunks), chunk_size)
    ]
    
    # Count in parallel
    with Pool(num_workers) as pool:
        results = pool.map(count_pairs, splits)
    
    # Merge all counters
    final_counts = Counter()
    for result in results:
        final_counts.update(result)
    
    return final_counts

def merge_pair(corpus_chunks: list[tuple[int]], pair: tuple[int, int], new_id: int) -> list[tuple[int]]:
    """
    Replace all the occurences of pair with new_id across all chunks

    eg: corpus_chunks = [(104,101,108,108,111)]
        pair = (108,108), new_id = 260 -> [(104,101,260,111)]
    """

    new_corpus = []

    for chunk in corpus_chunks:
        new_chunk = []
        i = 0
        while i < len(chunk):
            # check if the current position matches the pair
            if i < len(chunk) - 1 and chunk[i] == pair[0] and chunk[i+1] == pair[1]:
                new_chunk.append(new_id)
                i+=2 #skip both tokens of the pair
            else:
                new_chunk.append(chunk[i])
                i+=1
        new_corpus.append(tuple(new_chunk))

    return new_corpus



# Save ---------------------------------------------------

def save_tokenizer(vocab: dict, merges: list, config: dict, save_dir: str):
    
    os.makedirs(save_dir, exist_ok = True)

    # Save vocab: token_string -> id
    with open(f"{save_dir}/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii = False, indent = 2)

    # Save merges:one merge rule per line "id1 id2 new_id"
    with open(f"{save_dir}/merges.txt", "w", encoding="utf-8") as f:
        for (a, b), new_id in merges:
            f.write(f"{a} {b} {new_id}\n")

    # Save config
    with open(f"{save_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent = 2)


    print(f"Saved tokenizer at {save_dir}/")
    print(f"Vocab.json -> {len(vocab)} tokens")
    print(f"Merges.txt -> {len(merges)} merge rules")


# Main training loop -----------------------------------------

def train(
        corpus_path: str,
        vocab_size: int = 500,
        save_dir: str = "saved",
        max_lines: int = None
):
    print(f"loading corpus from {corpus_path}")
    text = read_corpus(corpus_path, max_lines=max_lines)
    print(f"{len(text):,} characters loaded")

    # Step 1: Pretokenize into string chunks
    print("\nPretokenizing...")
    string_chunks = pretokenize(text)
    print(f"{len(string_chunks):,} chunks")


    # Step 2: Build the base vocabulary
    vocab = build_base_vocab()
    print(f"Base Vocab size: {len(vocab)}")


    # Step 3: Now we convert each each chunk to bytes
    ## corpus_chunks is a list of tuples of byte integers
    # Adding this step cuz we need to create byte to vocab id mapping to prevent errors
    byte_to_vocab_id = {}
    for token_str, vocab_id in vocab.items():
        # for single byte tokens, map byte-value to vocab_id
        if len(token_str) == 1:
            byte_val = ord(token_str)
            if byte_val < 256: # its base byte token
                byte_to_vocab_id[byte_val] = vocab_id

    # NOW we do this step to convert the chunks to vocab ids and not raw bytes 
    def chunk_to_vocab_ids(chunk: str) -> tuple[int]:
        raw_bytes = chunk.encode("utf-8")
        return tuple(byte_to_vocab_id[b] for b in raw_bytes)
    

    # this stores something fucking thing, ok it stores corrected corpus chunks with proper vocab ids instead of raw bytes

    corpus_chunks = [chunk_to_vocab_ids(chunk) for chunk in string_chunks] 



    # now how many merges do we need
    num_merges = vocab_size - len(vocab)
    print(f"Target vocab size {vocab_size}")
    print(f"Merges to perform {num_merges}")

    if num_merges<=0:
        print("Error")
        return
    

    # Step 4: BPE merge loop, now the merges start, aka Magic of BPE

    merges = [] # list of ((a,b), new_id) - ordered, this order matters at encode time


    print("Starting BPE training...")

    for i in range(num_merges):

        # Count all adjacent pairs in current corpus
        # pair_counts = count_pairs(corpus_chunks)
        pair_counts = count_pairs_parallel(corpus_chunks, num_workers=4)

        if not pair_counts:
            print(f"No more pairs to merge at step {i}")
            break

        # Find most frequent pairs
        best_pair = max(pair_counts, key= lambda p: pair_counts[p])
        best_count = pair_counts[best_pair]

        #Assign new id to this merged token
        new_id = len(vocab)

        #what does this new token look like as a String? we need to find the string representation of both tokens

        id_to_token = {v:k for k,v in vocab.items()}
        new_token = id_to_token[best_pair[0]] + id_to_token[best_pair[1]]

        #Add that to vocab
        vocab[new_token] = new_id

        # Record this merge
        merges.append((best_pair, new_id))

        # Apply this merge to entire corpus
        corpus_chunks = merge_pair(corpus_chunks, best_pair, new_id)


        # Progress

        print(
            f"Merge {i+1:4d}/{num_merges} ! "
            f"({id_to_token[best_pair[0]]!r} + {id_to_token[best_pair[1]]!r})"
            f"-> {new_token!r} !"
            f"freq={best_count}"
            )
            
    print(f"\nTraining complete. Final vocab size: {len(vocab)}")


    # Step 5: Save everything
    config = {
        "vocab_size" : len(vocab),
        "num_merges" : len(merges),
        "special_tokens" : SPECIAL_TOKENS,
        "base_vocab_size" : 260
    }

    save_tokenizer (vocab, merges, config, save_dir)


# Entry Point -------------------------------------------
if __name__ == "__main__":
    train(
        corpus_path="./training_data/raw/openwebtext.txt",
        vocab_size=10000,
        save_dir="saved",
        max_lines=50000
    )