"""
Text utilities for challenge generation.

Includes:
- Cipher utilities (Caesar, Vigenère, Atbash, etc.)
- Plaintext pools for different languages
- Text manipulation helpers
"""

import random
import string
import re
from typing import List, Tuple

# ============================================================================
# Alphabet and character sets
# ============================================================================

ALPHABET = string.ascii_lowercase
ALPHABET_SET = set(ALPHABET)

# Noise characters for adversarial challenges
NOISE_ALPHABET = list("~^`|_[]{}<>/\\—–…•§±¤$%@#&*+=") + ['\u200b', '\u2009', '\u2002']  # includes thin/zero-width spaces

# ============================================================================
# Cipher utilities
# ============================================================================

def caesar_shift_char(ch: str, k: int) -> str:
    """Shift a single character by k positions in the alphabet."""
    if ch.lower() not in ALPHABET_SET:
        return ch
    a = ALPHABET.index(ch.lower())
    b = (a + k) % 26
    out = ALPHABET[b]
    return out.upper() if ch.isupper() else out


def caesar_encrypt(text: str, k: int) -> str:
    """Encrypt text using Caesar cipher with shift k."""
    return "".join(caesar_shift_char(ch, k) for ch in text)


def caesar_decrypt(text: str, k: int) -> str:
    """Decrypt Caesar cipher with shift k."""
    return caesar_encrypt(text, -k)


def atbash_char(ch: str) -> str:
    """Apply Atbash cipher to a single character (A↔Z, B↔Y, etc.)."""
    if ch.lower() not in ALPHABET_SET:
        return ch
    a = ALPHABET.index(ch.lower())
    b = 25 - a
    out = ALPHABET[b]
    return out.upper() if ch.isupper() else out


def atbash(text: str) -> str:
    """Apply Atbash cipher to text."""
    return "".join(atbash_char(c) for c in text)


def rot13(text: str) -> str:
    """Apply ROT13 cipher (Caesar with k=13)."""
    return caesar_encrypt(text, 13)


def vigenere_encrypt(plaintext: str, key: str) -> str:
    """
    Encrypt text using Vigenère cipher.
    
    Args:
        plaintext: Text to encrypt
        key: Keyword (only alphabetic chars used)
    
    Returns:
        Encrypted text
    """
    key = re.sub(r"[^a-z]", "", key.lower())
    if not key:
        raise ValueError("Key must contain alphabetic characters.")
    
    out = []
    j = 0
    for ch in plaintext:
        if ch.lower() in ALPHABET_SET:
            p = ALPHABET.index(ch.lower())
            k = ALPHABET.index(key[j % len(key)])
            c = ALPHABET[(p + k) % 26]
            out.append(c.upper() if ch.isupper() else c)
            j += 1
        else:
            out.append(ch)
    return "".join(out)


def vigenere_decrypt(ciphertext: str, key: str) -> str:
    """Decrypt Vigenère cipher."""
    key = re.sub(r"[^a-z]", "", key.lower())
    if not key:
        raise ValueError("Key must contain alphabetic characters.")
    
    out = []
    j = 0
    for ch in ciphertext:
        if ch.lower() in ALPHABET_SET:
            c = ALPHABET.index(ch.lower())
            k = ALPHABET.index(key[j % len(key)])
            p = ALPHABET[(c - k) % 26]
            out.append(p.upper() if ch.isupper() else p)
            j += 1
        else:
            out.append(ch)
    return "".join(out)


def autokey_encrypt(plaintext: str, primer: str) -> str:
    """
    Encrypt using Autokey cipher (Vigenère variant).
    
    The key is the primer followed by the plaintext itself.
    """
    key = primer + re.sub(r"[^a-z]", "", plaintext.lower())
    return vigenere_encrypt(plaintext, key)


def autokey_decrypt(ciphertext: str, primer: str) -> str:
    """Decrypt Autokey cipher."""
    key = re.sub(r"[^a-z]", "", primer.lower())
    out = []
    
    for ch in ciphertext:
        if ch.lower() in ALPHABET_SET:
            c = ALPHABET.index(ch.lower())
            k = ALPHABET.index(key[len(out) % len(key)] if len(out) < len(key) else key[-1])
            p = ALPHABET[(c - k) % 26]
            out.append(p.upper() if ch.isupper() else p)
            # Extend key with decrypted char
            key += p.lower()
        else:
            out.append(ch)
    
    return "".join(out)


# ============================================================================
# Text manipulation utilities
# ============================================================================

def inject_noise(s: str, noise_p: float = 0.1) -> str:
    """Inject random noise characters into text."""
    out = []
    for ch in s:
        out.append(ch)
        if random.random() < noise_p:
            out.append(random.choice(NOISE_ALPHABET))
    return "".join(out)


def remove_noise(s: str) -> str:
    """Remove noise characters from text."""
    return "".join(ch for ch in s if ch not in NOISE_ALPHABET)


def split_words_keep_punct(s: str) -> List[str]:
    """Split on spaces but keep punctuation attached to tokens."""
    return s.split(" ")


def extract_first_letters(text: str) -> str:
    """Extract first letter of each word (for acrostic puzzles)."""
    words = text.split()
    return "".join(w[0] if w else "" for w in words)


def extract_nth_letters(text: str, n: int) -> str:
    """Extract the nth letter of each word."""
    words = text.split()
    letters = []
    for w in words:
        if len(w) >= n:
            letters.append(w[n-1])
    return "".join(letters)


def extract_line_initials(text: str) -> str:
    """Extract first letter of each line (for acrostic poems)."""
    lines = text.strip().split("\n")
    return "".join(line.strip()[0] if line.strip() else "" for line in lines)


# ============================================================================
# Plaintext pools
# ============================================================================

PLAINTEXT_POOLS = {
    "english": [
        "from headquarters to arriving assets meet at point sixteen fifty-six with codeword delta",
        "deliver message at north gate at seventeen hundred hours bring credentials",
        "enemy moving eastward at dawn coordinates follow on secure channel",
        "rally point is behind old warehouse password is northern lights",
        "open channel three at midnight and await further instructions",
        "send reconnaissance team west eight kilometers and report findings",
        "use cipher two to confirm identity before code handoff",
        "package arrives tuesday extract team ready at safe house alpha",
        "abort mission immediately cover is compromised return to base",
        "target acquired proceeding with phase two of operation sunrise",
        "weather conditions favorable for drop zone charlie confirmation needed",
        "all units stand down until further notice radio silence in effect",
    ],
    "norwegian": [
        "fra hovedkvarteret til kommende talenter møt opp klokken seksten femtiseks med kodeord lille",
        "lever melding ved nord porten klokken sytten null null ta med bevis",
        "fienden beveger seg østover ved daggry koordinater følger på radio",
        "samlingspunkt er bak gamle lageret passord nordlys",
        "åpne kanal tre ved midnatt og vent på videre instruks",
        "send rekognoseringslag mot vest åtte kilometer og rapporter funn",
        "bruk siffer to for å bekrefte identitet før utlevering av kode",
        "pakken ankommer tirsdag hent laget klar ved trygt hus alfa",
        "avbryt oppdraget umiddelbart dekning er kompromittert returner til basen",
        "mål identifisert fortsetter med fase to av operasjon soloppgang",
    ],
}

# Word banks for generating random messages
WORD_BANKS = {
    "english": [
        "north", "south", "east", "west", "channel", "warehouse", "point", "codeword",
        "reconnaissance", "message", "gate", "coordinates", "dawn", "midnight",
        "mission", "observation", "signal", "masked", "simple", "secret",
        "target", "extraction", "phase", "operation", "unit", "team", "base",
    ],
    "norwegian": [
        "nord", "sør", "øst", "vest", "kanal", "lager", "punkt", "kodeord",
        "rekognosering", "melding", "porten", "koordinater", "daggry", "midnatt",
        "oppdrag", "observasjon", "signal", "maskert", "enkelt", "hemmelig",
        "mål", "ekstraksjon", "fase", "operasjon", "enhet", "lag", "base",
    ],
}

# Themed keyword pools for cipher keys
THEME_KEYWORDS = {
    "NORWAY_GEOGRAPHY": ["fjell", "fjord", "nord", "sor", "ost", "vest", "oslo", "bergen", "troms", "svalbard", "lillehammer"],
    "NORSE_MYTH": ["odin", "thor", "freya", "loki", "valhall", "yggdrasil", "ragnarok", "asgard", "midgard", "fenrir"],
    "NATURE": ["gran", "elg", "ulv", "mose", "bjorn", "hav", "isbre", "nordlys", "aurora", "glacier"],
    "COLORS": ["blue", "red", "green", "gold", "silver", "black", "white", "crimson", "azure", "violet"],
    "ELEMENTS": ["fire", "water", "earth", "wind", "storm", "thunder", "lightning", "frost", "shadow", "light"],
}


def get_plaintext_pool(language: str = "english") -> List[str]:
    """Get the plaintext pool for a language."""
    return PLAINTEXT_POOLS.get(language.lower(), PLAINTEXT_POOLS["english"])


def get_word_bank(language: str = "english") -> List[str]:
    """Get the word bank for a language."""
    return WORD_BANKS.get(language.lower(), WORD_BANKS["english"])


def random_plaintext(language: str = "english") -> str:
    """Get a random plaintext message."""
    pool = get_plaintext_pool(language)
    return random.choice(pool)


def random_words(n: int, language: str = "english") -> str:
    """Generate n random words from the word bank."""
    bank = get_word_bank(language)
    return " ".join(random.choice(bank) for _ in range(n))


def random_keyword(theme: str = None) -> str:
    """Get a random keyword from a theme."""
    if theme and theme in THEME_KEYWORDS:
        return random.choice(THEME_KEYWORDS[theme])
    # Random theme
    theme = random.choice(list(THEME_KEYWORDS.keys()))
    return random.choice(THEME_KEYWORDS[theme])
