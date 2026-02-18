# # # from train import chunk_to_bytes
# # # from pretokenize import pretokenize

# # # test_text = "hello world"
# # # chunks = pretokenize(test_text)
# # # for chunk in chunks:
# # #     bytes_tuple = chunk_to_bytes(chunk)
# # #     print(f"{repr(chunk):20} -> {bytes_tuple}")
# # with open("./training_data/raw/openwebtext.txt", "r", encoding="utf-8") as f:
# #     lines = [f.readline() for _ in range(1000)]
# #     text = "".join(lines)
    
# # # Check for control characters
# # control_char_count = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
# # print(f"Total chars: {len(text)}")
# # print(f"Control chars (excluding newline/tab): {control_char_count}")

# # # Show where \u001c appears
# # if '\u001c' in text:
# #     print(f"\nFound \\u001c (File Separator) character!")
# #     print(f"First occurrence at position: {text.index(chr(0x1c))}")
# #     # Show context around first occurrence
# #     pos = text.index(chr(0x1c))
# #     print(f"Context: {repr(text[max(0, pos-50):pos+50])}")
# # else:
# #     print("\nNo \\u001c found in first 1000 lines")



# from train import read_corpus, pretokenize, chunk_to_bytes, build_base_vocab, count_pairs

# # Load small sample
# text = read_corpus("training_data/raw/openwebtext.txt", max_lines=100)
# print(f"Sample text (first 200 chars): {repr(text[:200])}\n")

# # Pretokenize
# chunks = pretokenize(text)
# print(f"First 10 chunks: {chunks[:10]}\n")

# # Convert to bytes
# byte_chunks = [chunk_to_bytes(c) for c in chunks[:10]]
# print(f"First 10 byte chunks:")
# for i, (chunk, bytes_) in enumerate(zip(chunks[:10], byte_chunks)):
#     print(f"  {repr(chunk):20} → {bytes_}")

# # Build vocab and check what byte 28 (0x1c) maps to
# vocab = build_base_vocab()
# id_to_token = {v: k for k, v in vocab.items()}

# # Byte 28 is 0x1c (the control char you're seeing)
# byte_28_char = id_to_token[28 + 4]  # +4 offset for special tokens
# print(f"\nByte 28 (0x1c) maps to token ID {28+4}")
# print(f"Token string: {repr(byte_28_char)}")
# print(f"Is it actually \\u001c? {byte_28_char == chr(0x1c)}")

# # Now check: does byte 28 actually appear in your corpus?
# all_byte_chunks = [chunk_to_bytes(c) for c in chunks]
# byte_28_count = sum(chunk.count(28) for chunk in all_byte_chunks)
# print(f"\nByte 28 appears {byte_28_count} times in first 100 lines")


from train import read_corpus, pretokenize, chunk_to_bytes, build_base_vocab, count_pairs

text = read_corpus("training_data/raw/openwebtext.txt", max_lines=1000)
chunks = pretokenize(text)
corpus_chunks = [chunk_to_bytes(c) for c in chunks]

# Count pairs
pair_counts = count_pairs(corpus_chunks)

# Get top 10 most frequent pairs
top_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:10]

vocab = build_base_vocab()
id_to_token = {v: k for k, v in vocab.items()}

print("Top 10 most frequent pairs:")
for pair, count in top_pairs:
    token_a = id_to_token[pair[0]]
    token_b = id_to_token[pair[1]]
    print(f"  ({repr(token_a)} + {repr(token_b)}) → freq={count}")