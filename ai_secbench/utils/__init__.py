"""
Utility modules for AI-SecBench.
"""

from ai_secbench.utils.text import (
    # Cipher utilities
    caesar_encrypt,
    caesar_decrypt,
    vigenere_encrypt,
    vigenere_decrypt,
    atbash,
    rot13,
    autokey_encrypt,
    autokey_decrypt,
    # Text utilities
    inject_noise,
    remove_noise,
    extract_first_letters,
    extract_nth_letters,
    extract_line_initials,
    # Plaintext pools
    get_plaintext_pool,
    get_word_bank,
    random_plaintext,
    random_words,
    random_keyword,
    PLAINTEXT_POOLS,
    WORD_BANKS,
    THEME_KEYWORDS,
)

__all__ = [
    # Ciphers
    "caesar_encrypt",
    "caesar_decrypt",
    "vigenere_encrypt",
    "vigenere_decrypt",
    "atbash",
    "rot13",
    "autokey_encrypt",
    "autokey_decrypt",
    # Text
    "inject_noise",
    "remove_noise",
    "extract_first_letters",
    "extract_nth_letters",
    "extract_line_initials",
    # Pools
    "get_plaintext_pool",
    "get_word_bank",
    "random_plaintext",
    "random_words",
    "random_keyword",
    "PLAINTEXT_POOLS",
    "WORD_BANKS",
    "THEME_KEYWORDS",
]
