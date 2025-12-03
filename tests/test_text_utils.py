"""
Tests for text utilities and cipher operations.
"""
import pytest
from ai_secbench.utils.text import (
    normalize_text,
    normalize_for_grading,
    remove_noise,
    inject_noise,
    caesar_encrypt,
    caesar_decrypt,
    vigenere_encrypt,
    vigenere_decrypt,
    atbash,
    rot13,
    compute_text_similarity,
    get_alphabet,
    ALPHABETS,
    NOISE_CHARS,
)


class TestAlphabets:
    """Tests for alphabet handling."""
    
    def test_english_alphabet(self):
        assert get_alphabet("en") == "abcdefghijklmnopqrstuvwxyz"
        assert len(get_alphabet("en")) == 26
    
    def test_norwegian_alphabet(self):
        no_alpha = get_alphabet("no")
        assert "æ" in no_alpha
        assert "ø" in no_alpha
        assert "å" in no_alpha
        assert len(no_alpha) == 29
    
    def test_default_alphabet(self):
        assert get_alphabet("unknown") == ALPHABETS["en"]


class TestNormalization:
    """Tests for text normalization."""
    
    def test_normalize_text_lowercase(self):
        assert normalize_text("HELLO World") == "hello world"
    
    def test_normalize_text_preserve_case(self):
        assert normalize_text("HELLO World", preserve_case=True) == "HELLO World"
    
    def test_normalize_text_whitespace(self):
        assert normalize_text("hello   world\n\t") == "hello world"
    
    def test_normalize_for_grading(self):
        assert normalize_for_grading("  HELLO, World!  ") == "hello, world!"
    
    def test_normalize_for_grading_unicode(self):
        # Norwegian characters should be preserved
        assert "æ" in normalize_for_grading("Blåbær")
        assert "ø" in normalize_for_grading("Grønn")


class TestNoiseOperations:
    """Tests for noise injection and removal."""
    
    def test_inject_noise_deterministic(self):
        text = "hello world"
        noisy1 = inject_noise(text, 0.5, seed=42)
        noisy2 = inject_noise(text, 0.5, seed=42)
        assert noisy1 == noisy2
    
    def test_inject_noise_adds_characters(self):
        text = "hello"
        noisy = inject_noise(text, 0.9, seed=42)
        assert len(noisy) > len(text)
    
    def test_remove_noise(self):
        text = "hello"
        noisy = inject_noise(text, 0.5, seed=42)
        cleaned = remove_noise(noisy)
        assert cleaned == text
    
    def test_noise_roundtrip(self):
        original = "test message with spaces"
        noisy = inject_noise(original, 0.3, seed=123)
        cleaned = remove_noise(noisy)
        assert cleaned == original


class TestCaesarCipher:
    """Tests for Caesar cipher operations."""
    
    def test_caesar_encrypt_simple(self):
        assert caesar_encrypt("abc", 1) == "bcd"
        assert caesar_encrypt("xyz", 1) == "yza"
    
    def test_caesar_encrypt_rot13(self):
        assert caesar_encrypt("hello", 13) == "uryyb"
    
    def test_caesar_decrypt(self):
        encrypted = caesar_encrypt("hello", 7)
        assert caesar_decrypt(encrypted, 7) == "hello"
    
    def test_caesar_preserves_case(self):
        assert caesar_encrypt("Hello World", 3) == "Khoor Zruog"
    
    def test_caesar_preserves_non_alpha(self):
        assert caesar_encrypt("hello, world!", 1) == "ifmmp, xpsme!"
    
    def test_caesar_norwegian(self):
        no_alpha = get_alphabet("no")
        # Test with Norwegian alphabet
        encrypted = caesar_encrypt("å", 1, no_alpha)
        decrypted = caesar_decrypt(encrypted, 1, no_alpha)
        assert decrypted == "å"
    
    def test_caesar_full_cycle(self):
        # Shift by 26 should return original
        assert caesar_encrypt("test", 26) == "test"


class TestVigenereCipher:
    """Tests for Vigenère cipher operations."""
    
    def test_vigenere_encrypt_simple(self):
        # Classic example: ATTACKATDAWN with key LEMON
        plaintext = "attackatdawn"
        key = "lemon"
        encrypted = vigenere_encrypt(plaintext, key)
        assert encrypted == "lxfopvefrnhr"
    
    def test_vigenere_decrypt(self):
        plaintext = "hello world"
        key = "key"
        encrypted = vigenere_encrypt(plaintext, key)
        decrypted = vigenere_decrypt(encrypted, key)
        assert decrypted == plaintext
    
    def test_vigenere_preserves_case(self):
        encrypted = vigenere_encrypt("Hello", "key")
        assert encrypted[0].isupper()
    
    def test_vigenere_preserves_spaces(self):
        encrypted = vigenere_encrypt("hello world", "key")
        assert " " in encrypted
    
    def test_vigenere_empty_key_raises(self):
        with pytest.raises(ValueError):
            vigenere_encrypt("test", "")
    
    def test_vigenere_roundtrip_norwegian(self):
        no_alpha = get_alphabet("no")
        plaintext = "hei på deg"
        key = "nøkkel"
        encrypted = vigenere_encrypt(plaintext, key, no_alpha)
        decrypted = vigenere_decrypt(encrypted, key, no_alpha)
        assert decrypted == plaintext


class TestAtbash:
    """Tests for Atbash cipher."""
    
    def test_atbash_simple(self):
        assert atbash("a") == "z"
        assert atbash("z") == "a"
        assert atbash("m") == "n"
    
    def test_atbash_word(self):
        assert atbash("hello") == "svool"
    
    def test_atbash_is_self_inverse(self):
        text = "test message"
        assert atbash(atbash(text)) == text
    
    def test_atbash_preserves_case(self):
        assert atbash("Hello") == "Svool"


class TestRot13:
    """Tests for ROT13."""
    
    def test_rot13_basic(self):
        assert rot13("hello") == "uryyb"
    
    def test_rot13_is_self_inverse(self):
        text = "test message"
        assert rot13(rot13(text)) == text


class TestTextSimilarity:
    """Tests for text similarity computation."""
    
    def test_identical_texts(self):
        assert compute_text_similarity("hello", "hello") == 1.0
    
    def test_completely_different(self):
        sim = compute_text_similarity("aaa", "zzz")
        assert sim < 0.5
    
    def test_similar_texts(self):
        sim = compute_text_similarity("hello world", "hello worlds")
        assert sim > 0.8
    
    def test_empty_strings(self):
        assert compute_text_similarity("", "") == 1.0
        assert compute_text_similarity("hello", "") == 0.0
    
    def test_case_insensitive(self):
        # Should normalize before comparing
        sim = compute_text_similarity("HELLO", "hello")
        assert sim == 1.0
