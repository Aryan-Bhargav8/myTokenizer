# Check what's actually in the file you just downloaded
with open("training_data/raw/openwebtext.txt", "rb") as f:  # note: 'rb' = read bytes
    raw_bytes = f.read(10000)  # first 10k bytes

# Count byte 28 (which is \x1c)
byte_28_count = raw_bytes.count(28)
print(f"Byte 28 (\\x1c) appears {byte_28_count} times in first 10k bytes")

# Show first occurrence
if 28 in raw_bytes:
    pos = raw_bytes.index(28)
    print(f"\nFirst \\x1c at position {pos}")
    print(f"Context (bytes): {raw_bytes[max(0,pos-20):pos+20]}")
    print(f"Context (string): {repr(raw_bytes[max(0,pos-20):pos+20].decode('utf-8', errors='replace'))}")