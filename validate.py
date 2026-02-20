import os
from tokenizer import Tokenizer
from pretokenize import pretokenize


def test_round_trip(tokenizer: Tokenizer, test_cases: list[str]) -> bool:
    """
    Test that encode → decode returns the original text.
    This is the most critical test - if this fails, tokenizer is broken.
    """
    print("=" * 80)
    print("TEST 1: Round-Trip Encoding/Decoding")
    print("=" * 80)
    
    all_passed = True
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Original text: {repr(text)}")
        
        ids = tokenizer.encode(text)
        print(f"Encoded to {len(ids)} tokens: {ids}")
        
        # Show first few actual token strings
        token_strings = [tokenizer.id_to_token.get(id, "<UNK>") for id in ids[:10]]
        print(f"Token strings: {[repr(t) for t in token_strings]}{'...' if len(ids) > 10 else ''}")
        
        decoded = tokenizer.decode(ids)
        print(f"Decoded text:  {repr(decoded)}")
        
        passed = (text == decoded)
        
        if not passed:
            all_passed = False
            print(f"❌ MISMATCH DETECTED")
            print(f"  Length: original={len(text)}, decoded={len(decoded)}")
            # Show where they differ
            for j, (c1, c2) in enumerate(zip(text, decoded)):
                if c1 != c2:
                    print(f"  First diff at position {j}: {repr(c1)} vs {repr(c2)}")
                    break
        else:
            print(f"✓ PASS - Perfect round-trip")
    
    print(f"\n{'='*80}")
    print(f"{'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}: Round-trip test")
    print(f"{'='*80}")
    return all_passed


def test_compression_metrics(tokenizer: Tokenizer, corpus_path: str, max_chars: int = 100000) -> dict:
    """
    Measure tokenizer compression efficiency.
    Good tokenizers should achieve 3-4 chars/token for English.
    """
    print("\n" + "=" * 80)
    print("TEST 2: Compression Metrics")
    print("=" * 80)
    
    # Load sample from corpus
    print(f"\nLoading {max_chars:,} characters from {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read(max_chars)
    
    print(f"Sample text (first 200 chars):")
    print(f"  {repr(text[:200])}...")
    
    # Tokenize
    print(f"\nTokenizing...")
    ids = tokenizer.encode(text)
    
    # Calculate metrics
    num_chars = len(text)
    num_tokens = len(ids)
    chars_per_token = num_chars / num_tokens if num_tokens > 0 else 0
    bytes_per_token = len(text.encode('utf-8')) / num_tokens if num_tokens > 0 else 0
    
    print(f"\n--- Results ---")
    print(f"Characters:         {num_chars:,}")
    print(f"Bytes (UTF-8):      {len(text.encode('utf-8')):,}")
    print(f"Tokens:             {num_tokens:,}")
    print(f"Chars per token:    {chars_per_token:.2f}")
    print(f"Bytes per token:    {bytes_per_token:.2f}")
    
    # Show sample tokenization
    sample_text = text[:100]
    sample_ids = tokenizer.encode(sample_text)
    sample_tokens = [tokenizer.id_to_token.get(id, "<UNK>") for id in sample_ids]
    print(f"\nSample tokenization of first 100 chars:")
    print(f"  Input:  {repr(sample_text)}")
    print(f"  Tokens: {sample_tokens}")
    print(f"  Count:  {len(sample_ids)} tokens")
    
    # Evaluation
    print(f"\n--- Evaluation ---")
    if chars_per_token >= 3.5:
        print("✓ EXCELLENT: Compression is very efficient (≥3.5 chars/token)")
        status = "excellent"
    elif chars_per_token >= 3.0:
        print("✓ GOOD: Compression is efficient (3.0-3.5 chars/token)")
        status = "good"
    elif chars_per_token >= 2.0:
        print("⚠ OK: Compression is acceptable (2.0-3.0 chars/token)")
        status = "ok"
    else:
        print("✗ POOR: Compression is inefficient (<2.0 chars/token)")
        status = "poor"
    
    print(f"{'='*80}")
    
    return {
        "num_chars": num_chars,
        "num_tokens": num_tokens,
        "chars_per_token": chars_per_token,
        "bytes_per_token": bytes_per_token,
        "status": status
    }


def test_special_tokens(tokenizer: Tokenizer) -> bool:
    """
    Test that special tokens work correctly.
    """
    print("\n" + "=" * 80)
    print("TEST 3: Special Tokens")
    print("=" * 80)
    
    all_passed = True
    
    text = "Hello, world!"
    print(f"\nTest text: {repr(text)}")
    
    # Encode without special tokens
    print(f"\n--- Without special tokens ---")
    ids_without = tokenizer.encode(text, add_special_tokens=False)
    print(f"Token IDs: {ids_without}")
    print(f"Count: {len(ids_without)} tokens")
    
    # Encode with special tokens
    print(f"\n--- With special tokens ---")
    ids_with = tokenizer.encode(text, add_special_tokens=True)
    print(f"Token IDs: {ids_with}")
    print(f"Count: {len(ids_with)} tokens")
    
    # Check BOS and EOS are added
    bos_id = tokenizer.special_tokens["<BOS>"]
    eos_id = tokenizer.special_tokens["<EOS>"]
    
    print(f"\nSpecial token IDs:")
    print(f"  <BOS>: {bos_id}")
    print(f"  <EOS>: {eos_id}")
    print(f"  <PAD>: {tokenizer.special_tokens['<PAD>']}")
    print(f"  <UNK>: {tokenizer.special_tokens['<UNK>']}")
    
    print(f"\nChecking positions:")
    print(f"  First token: {ids_with[0]} (expected <BOS>={bos_id})")
    print(f"  Last token:  {ids_with[-1]} (expected <EOS>={eos_id})")
    
    if ids_with[0] == bos_id and ids_with[-1] == eos_id:
        print(f"✓ <BOS> and <EOS> correctly added")
    else:
        print(f"✗ FAILED: Special tokens not added correctly")
        all_passed = False
    
    # Test decoding with skip_special_tokens
    print(f"\n--- Decoding test ---")
    decoded_without_skip = tokenizer.decode(ids_with, skip_special_tokens=False)
    decoded_with_skip = tokenizer.decode(ids_with, skip_special_tokens=True)
    
    print(f"Decoded (keep special):   {repr(decoded_without_skip)}")
    print(f"Decoded (skip special):   {repr(decoded_with_skip)}")
    print(f"Original text:            {repr(text)}")
    
    if decoded_with_skip == text:
        print(f"✓ skip_special_tokens=True works correctly")
    else:
        print(f"✗ FAILED: skip_special_tokens not working")
        all_passed = False
    
    print(f"\n{'='*80}")
    print(f"{'✓ PASSED' if all_passed else '✗ FAILED'}: Special tokens test")
    print(f"{'='*80}")
    return all_passed


def test_edge_cases(tokenizer: Tokenizer) -> bool:
    """
    Test edge cases that often break tokenizers.
    """
    print("\n" + "=" * 80)
    print("TEST 4: Edge Cases")
    print("=" * 80)
    
    edge_cases = [
        ("Empty string", ""),
        ("Single space", " "),
        ("Multiple spaces", "   "),
        ("Single newline", "\n"),
        ("Multiple newlines", "\n\n\n"),
        ("Tab character", "\t"),
        ("Single char", "a"),
        ("Punctuation", "!@#$%^&*()"),
        ("Numbers", "123456789"),
        ("Leading spaces", "   leading spaces"),
        ("Trailing spaces", "trailing spaces   "),
        ("Mid spaces", "mid   dle   spaces"),
    ]
    
    all_passed = True
    
    for i, (name, text) in enumerate(edge_cases, 1):
        print(f"\n--- Case {i}: {name} ---")
        print(f"Input: {repr(text)}")
        
        try:
            ids = tokenizer.encode(text)
            print(f"Token IDs: {ids}")
            
            decoded = tokenizer.decode(ids)
            print(f"Decoded: {repr(decoded)}")
            
            passed = (text == decoded)
            
            if passed:
                print(f"✓ PASS")
            else:
                all_passed = False
                print(f"✗ FAIL - Mismatch!")
                print(f"  Expected length: {len(text)}")
                print(f"  Got length:      {len(decoded)}")
        except Exception as e:
            all_passed = False
            print(f"✗ EXCEPTION: {type(e).__name__}: {e}")
    
    print(f"\n{'='*80}")
    print(f"{'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}: Edge cases test")
    print(f"{'='*80}")
    return all_passed


def test_unicode_handling(tokenizer: Tokenizer) -> bool:
    """
    Test handling of various unicode characters and scripts.
    """
    print("\n" + "=" * 80)
    print("TEST 5: Unicode Handling")
    print("=" * 80)
    
    unicode_cases = [
        ("ASCII", "Hello, world!"),
        ("Accents", "Héllo, wörld!"),
        ("Chinese", "你好世界"),
        ("Japanese", "こんにちは"),
        ("Korean", "안녕하세요"),
        ("Russian", "Привет мир"),
        ("Arabic", "مرحبا بالعالم"),
        ("Emojis", "🌍🌎🌏"),
        ("French", "café"),
        ("Diaeresis", "naïve"),
    ]
    
    all_passed = True
    
    for i, (name, text) in enumerate(unicode_cases, 1):
        print(f"\n--- Case {i}: {name} ---")
        print(f"Input: {repr(text)}")
        print(f"Bytes: {len(text.encode('utf-8'))} bytes, {len(text)} chars")
        
        try:
            ids = tokenizer.encode(text)
            print(f"Tokens: {len(ids)} tokens")
            print(f"Token IDs: {ids}")
            
            decoded = tokenizer.decode(ids)
            print(f"Decoded: {repr(decoded)}")
            
            passed = (text == decoded)
            
            if passed:
                ratio = len(text) / len(ids) if len(ids) > 0 else 0
                print(f"✓ PASS ({ratio:.2f} chars/token)")
            else:
                all_passed = False
                print(f"✗ FAIL - Mismatch!")
        except Exception as e:
            all_passed = False
            print(f"✗ EXCEPTION: {type(e).__name__}: {e}")
    
    print(f"\n{'='*80}")
    print(f"{'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}: Unicode handling test")
    print(f"{'='*80}")
    return all_passed


def test_consistency(tokenizer: Tokenizer) -> bool:
    """
    Test that tokenizing the same text multiple times gives same results.
    """
    print("\n" + "=" * 80)
    print("TEST 6: Consistency")
    print("=" * 80)
    
    text = "The quick brown fox jumps over the lazy dog."
    print(f"\nTest text: {repr(text)}")
    print(f"Encoding same text 5 times...\n")
    
    # Encode same text 5 times
    results = []
    for i in range(5):
        ids = tokenizer.encode(text)
        results.append(ids)
        print(f"Run {i+1}: {ids}")
    
    # All should be identical
    all_same = all(r == results[0] for r in results)
    
    print(f"\n--- Check ---")
    if all_same:
        print(f"✓ All runs produced identical results")
        print(f"✓ Tokenization is deterministic")
    else:
        print(f"✗ FAILED: Runs produced different results!")
        print(f"This indicates non-deterministic behavior - serious bug!")
    
    print(f"\n{'='*80}")
    print(f"{'✓ PASSED' if all_same else '✗ FAILED'}: Consistency test")
    print(f"{'='*80}")
    return all_same


def run_all_tests(tokenizer_path: str = "saved/", corpus_path: str = "./training_data/raw/openwebtext.txt"):
    """
    Run complete validation suite.
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "TOKENIZER VALIDATION SUITE")
    print("=" * 80)
    print(f"Tokenizer path: {tokenizer_path}")
    print(f"Corpus path:    {corpus_path}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = Tokenizer.from_pretrained(tokenizer_path)
    print(f"✓ Loaded successfully")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    print(f"  Special tokens: {list(tokenizer.special_tokens.keys())}")
    
    # Define test cases
    basic_test_cases = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "I don't think so...",
        "This is a test of the tokenizer.",
        "Multiple sentences. Like this one. And this one!",
        "Numbers: 1234567890",
        "Special chars: !@#$%^&*()_+-=[]{}|;:,.<>?",
    ]
    
    # Run all tests
    results = {}
    
    results["round_trip"] = test_round_trip(tokenizer, basic_test_cases)
    
    if os.path.exists(corpus_path):
        results["compression"] = test_compression_metrics(tokenizer, corpus_path)
    else:
        print(f"\n⚠ Skipping compression test - corpus not found at {corpus_path}")
        results["compression"] = {"status": "skipped", "chars_per_token": 0}
    
    results["special_tokens"] = test_special_tokens(tokenizer)
    results["edge_cases"] = test_edge_cases(tokenizer)
    results["unicode"] = test_unicode_handling(tokenizer)
    results["consistency"] = test_consistency(tokenizer)
    
    # Summary
    print("\n" + "=" * 80)
    print(" " * 35 + "SUMMARY")
    print("=" * 80)
    
    print(f"\nTest Results:")
    print(f"  Round-trip:      {'✓ PASS' if results['round_trip'] else '✗ FAIL'}")
    print(f"  Special tokens:  {'✓ PASS' if results['special_tokens'] else '✗ FAIL'}")
    print(f"  Edge cases:      {'✓ PASS' if results['edge_cases'] else '✗ FAIL'}")
    print(f"  Unicode:         {'✓ PASS' if results['unicode'] else '✗ FAIL'}")
    print(f"  Consistency:     {'✓ PASS' if results['consistency'] else '✗ FAIL'}")
    
    if results['compression']['status'] != 'skipped':
        print(f"  Compression:     {results['compression']['chars_per_token']:.2f} chars/token ({results['compression']['status']})")
    
    passed = sum([
        results["round_trip"],
        results["special_tokens"],
        results["edge_cases"],
        results["unicode"],
        results["consistency"]
    ])
    total = 5
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "=" * 80)
        print("✓✓✓ ALL TESTS PASSED - Tokenizer is production ready! ✓✓✓")
        print("=" * 80)
        return True
    else:
        print("\n" + "=" * 80)
        print("✗✗✗ SOME TESTS FAILED - Review failures above ✗✗✗")
        print("=" * 80)
        return False


if __name__ == "__main__":
    run_all_tests(
        tokenizer_path="saved/",
        corpus_path="./training_data/raw/openwebtext.txt"
    )