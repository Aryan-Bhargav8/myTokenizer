import json
from pretokenize import pretokenize


class Tokenizer:
    def __init__(self, vocab:dict, merges: list, special_tokens:dict):
        """
        Initialize tokenizer with trained vocabulary and merge rules.

        vocab: token_string -> id mapping
        merges: list of ((id_a, id_b), new_id) in order
        special_tokens: dict of special tokens names -> ids
        """


        self.vocab = vocab
        self.id_to_token = {v:k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens


        # Building merge lookup for faster encoding
        # Maps (id_a, id_b) -> new_id
        self.merge_map = {pair:new_id for pair, new_id in merges}


    @classmethod
    def from_pretrained(cls, model_dir: str):
        """
        Load a trained tokenizer from disk.

        Usage:
            tokenizer = Tokenizer.from_pretained("saved/")
        """

        #load vocab

        with open(f"{model_dir}/vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)

        
        #load merges
        merges = []
        with open(f"{model_dir}/merges.txt", "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    id_a, id_b, new_id = map(int, parts)
                    merges.append(((id_a,id_b), new_id))

        #load config
        with open(f"{model_dir}/config.json","r") as f:
            config = json.load(f)

        special_tokens = config["special_tokens"]

        return cls(vocab, merges, special_tokens)
    


    def _text_to_ids(self, text: str) -> list[int]:
        """
        Convert text to base token IDs (before merging)
        This maps each byte to its vocab ID
        """

        # Build byte -> vocab_id mapping

        byte_to_id = {}
        for token_str, vocab_id in self.vocab.items():
            if len(token_str) == 1:
                byte_val = ord(token_str)
                if byte_val < 256:
                    byte_to_id[byte_val] = vocab_id

        # Convert text bytes to vocab ids
        raw_bytes = text.encode("utf-8")
        return [byte_to_id[b] for b in raw_bytes]
    

    def _apply_merges(self, ids: list[int]) -> list[int]:
        """
        Apply BPE merges to a sequence of token IDs.
        Merges are applied in the order they were learned during training.
        """

        # Keep applying merges until no more can be applied
        while len(ids) >= 2:
            #find all pairs in current sequence
            pairs = [(ids[i], ids[i+1], i) for i in range(len(ids) - 1)]

            #find the pairs that should be merged first (the one that was learned earliest in training)
            best_pair = None
            best_idx = None
            best_merge_position = len(self.merges) # here it starts with "not found"


            for pair_a, pair_b, pos in pairs:
                pair = (pair_a, pair_b)
                if pair in self.merge_map:
                    # find when this merge was learned
                    merge_position = next(
                        i for i,(p, _) in enumerate(self.merges) if p == pair
                    )

                    #earlier merges have priority
                    if merge_position < best_merge_position:
                        best_merge_position = merge_position
                        best_pair = pair
                        best_idx = pos

            
            # No more merges possible
            if best_pair is None:
                break

            # Apply the merge
            new_id = self.merge_map[best_pair]
            ids = ids[:best_idx] + [new_id] + ids[best_idx+2:]

        return ids
    

    def encode(self, text:str, add_special_tokens: bool = False) -> list[int]:
        """
        Encode text into token IDs

        Args:
        - text : input text string
        - add special tokens: if true, add <BOS> <EOS> tokens

        Returns:
        - List of token ids
        """

        # step1: Pretokenize the text
        chunks = pretokenize(text)

        #step2: Encode each chunk independently
        all_ids = []
        for chunk in chunks:
            # convert chunk to base token IDs
            chunk_ids = self._text_to_ids(chunk)
            # Apply BPE merges
            merged_ids = self._apply_merges(chunk_ids)
            all_ids.extend(merged_ids)

        # step3: add special tokens if requested
        if add_special_tokens:
            bos_id = self.special_tokens["<BOS>"]
            eos_id = self.special_tokens["<EOS>"]
            all_ids = [bos_id] + all_ids + [eos_id]

        return all_ids
    


    def decode(self, ids: list[int], skip_special_tokens:bool = True) -> str:
        """
        Decode token IDs back into text
        
        ARGS:
        - ids: List of token ids
        - skip special tokens: if true then remove special tokens from output
        
        Return:
        - Decoded text string
        """

        #Filter special tokens if requested
        if skip_special_tokens:
            special_ids = set(self.special_tokens.values())
            ids = [id for id in ids if id not in special_ids]


        # Map each ID to its token string
        tokens = []
        for id in ids:
            if id in self.id_to_token:
                tokens.append(self.id_to_token[id])
            else:
                # Unknown ID
                tokens.append(self.id_to_token[self.special_tokens["<UNK>"]])

        # Concatenate all token strings
        text = "".join(tokens)
        # text = ",".join(tokens) # for testing purpose


        # Convert from latin-1 bytes back to utf-8 text
        ## Remember: our vocab stores bytes as latin-1 strings

        try:
            # encode as latin-1 to get raw bytes, then decode as utf-8
            text_bytes = text.encode("latin-1")
            return text_bytes.decode("utf-8", errors="replace")
        except:
            return text # fallback if conversion fails
        


    
    def get_vocab_size(self) -> int:
        """Return vocabulary size"""
        return len(self.vocab)
    

    def token_to_id(self, token:str) -> int:
        """Get the ID for a specific token."""
        return self.vocab.get(token, self.special_tokens["<UNK>"])
    

    def id_to_token_str(self, id:int) -> str:
        """Get the token string for a specific ID."""



# Usage Example --------------------------------------------

if __name__ == "__main__" :
    #load trained tokenizer

    tokenizer = Tokenizer.from_pretrained("saved/")

    print(f"Loaded tokenizer with vocab size: {tokenizer.get_vocab_size()}\n")

    # Test encoding
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "I don't think so...",
        "만나서 반가워요",  # Korean
    ]

    for text in test_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)

        print(f"Original:  {repr(text)}")
        print(f"Token IDs: {ids}")
        print(f"Decoded:   {repr(decoded)}")
        print(f"Match:     {text == decoded}")
        print(f"Tokens:    {len(ids)}")
        print()

    # print("Type a sentence to encode/decode.")
    # print("Type 0 and press Enter to exit.\n")

    # while True:
    #     # Take input
    #     text = input("Enter text: ")

    #     # Exit condition
    #     if text.strip() == "0":
    #         print("\nExiting tokenizer tester. Goodbye!")
    #         break

    #     # Encode
    #     ids = tokenizer.encode(text)

    #     # Decode
    #     decoded = tokenizer.decode(ids)

    #     # Display results
    #     print("\n--- Result ---")
    #     print(f"Original : {repr(text)}")
    #     print(f"Token IDs: {ids}")
    #     print(f"Decoded  : {repr(decoded)}")
    #     print(f"Match    : {text == decoded}")
    #     print(f"Tokens   : {len(ids)}")
    #     print("------------\n")