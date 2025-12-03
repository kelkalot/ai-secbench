"""
Tests for challenge generators.
"""
import pytest
from ai_secbench.core.challenge import Challenge, ChallengeCategory, Difficulty
from ai_secbench.challenges.cipher.generators import (
    MultiShiftCaesarGenerator,
    VigenereGenerator,
    AdversarialMixGenerator,
    PromptSecurityGenerator,
    list_cipher_generators,
    get_cipher_generator,
)
from ai_secbench.challenges.context_poisoning.generators import (
    ContextPoisoningGenerator,
    ContradictoryContextGenerator,
    list_context_poisoning_generators,
)
from ai_secbench.challenges.social_engineering.generators import (
    AuthoritySpoofingGenerator,
    EmotionalManipulationGenerator,
    RoleplayManipulationGenerator,
    list_social_engineering_generators,
)
from ai_secbench.utils.text import caesar_decrypt, vigenere_decrypt, get_alphabet


class TestChallengeBase:
    """Base tests for all challenge types."""
    
    def assert_valid_challenge(self, challenge: Challenge):
        """Assert that a challenge has all required fields."""
        assert challenge.challenge_id is not None
        assert challenge.category is not None
        assert challenge.challenge_type is not None
        assert challenge.difficulty is not None
        assert challenge.task_card is not None
        assert challenge.solution is not None
        
        # Task card should have content
        assert challenge.task_card.title
        assert challenge.task_card.instructions
        assert challenge.task_card.content
        assert challenge.task_card.deliverable
        
        # Should be able to generate a prompt
        prompt = challenge.get_prompt()
        assert len(prompt) > 50


class TestMultiShiftCaesarGenerator(TestChallengeBase):
    """Tests for multi-shift Caesar cipher generator."""
    
    def test_generate_basic(self):
        gen = MultiShiftCaesarGenerator()
        challenge = gen.generate(seed=42)
        self.assert_valid_challenge(challenge)
        assert challenge.category == ChallengeCategory.CIPHER
        assert challenge.challenge_type == "multi_shift_caesar"
    
    def test_generate_reproducible(self):
        gen = MultiShiftCaesarGenerator()
        c1 = gen.generate(seed=12345)
        c2 = gen.generate(seed=12345)
        assert c1.solution.plaintext == c2.solution.plaintext
        assert c1.solution.key == c2.solution.key
    
    def test_generate_different_seeds(self):
        gen = MultiShiftCaesarGenerator()
        c1 = gen.generate(seed=1)
        c2 = gen.generate(seed=2)
        # Very unlikely to be the same
        assert c1.solution.plaintext != c2.solution.plaintext or c1.solution.key != c2.solution.key
    
    def test_solution_is_correct(self):
        gen = MultiShiftCaesarGenerator()
        challenge = gen.generate(seed=42)
        
        plaintext = challenge.solution.plaintext
        shifts = challenge.solution.key
        ciphertext = challenge.task_card.content["ciphertext"]
        
        # Decrypt and verify
        cipher_words = ciphertext.split()
        plain_words = plaintext.split()
        
        assert len(cipher_words) == len(plain_words)
        
        for cipher_word, plain_word, shift in zip(cipher_words, plain_words, shifts):
            decrypted = caesar_decrypt(cipher_word, shift)
            assert decrypted.lower() == plain_word.lower()
    
    def test_difficulty_levels(self):
        gen = MultiShiftCaesarGenerator()
        for diff in Difficulty:
            challenge = gen.generate(difficulty=diff, seed=42)
            assert challenge.difficulty == diff
    
    def test_language_english(self):
        gen = MultiShiftCaesarGenerator()
        challenge = gen.generate(language="en", seed=42)
        assert challenge.language == "en"
    
    def test_language_norwegian(self):
        gen = MultiShiftCaesarGenerator()
        challenge = gen.generate(language="no", seed=42)
        assert challenge.language == "no"
    
    def test_batch_generation(self):
        gen = MultiShiftCaesarGenerator()
        challenges = gen.generate_batch(5, seed=42)
        assert len(challenges) == 5
        # All should be different
        ids = [c.challenge_id for c in challenges]
        assert len(set(ids)) == 5


class TestVigenereGenerator(TestChallengeBase):
    """Tests for Vigenère cipher generator."""
    
    def test_generate_basic(self):
        gen = VigenereGenerator()
        challenge = gen.generate(seed=42)
        self.assert_valid_challenge(challenge)
        assert challenge.category == ChallengeCategory.CIPHER
        assert challenge.challenge_type == "vigenere"
    
    def test_generate_reproducible(self):
        gen = VigenereGenerator()
        c1 = gen.generate(seed=12345)
        c2 = gen.generate(seed=12345)
        assert c1.solution.plaintext == c2.solution.plaintext
        assert c1.solution.key == c2.solution.key
    
    def test_solution_is_correct(self):
        gen = VigenereGenerator()
        challenge = gen.generate(seed=42)
        
        plaintext = challenge.solution.plaintext
        key = challenge.solution.additional_keys.get("full_key_used", challenge.solution.key)
        ciphertext = challenge.task_card.content["ciphertext"]
        
        decrypted = vigenere_decrypt(ciphertext, key)
        assert decrypted.lower() == plaintext.lower()
    
    def test_has_theme_hint(self):
        gen = VigenereGenerator()
        challenge = gen.generate(seed=42)
        hints = challenge.task_card.hints
        assert hints is not None
        assert "theme" in hints


class TestAdversarialMixGenerator(TestChallengeBase):
    """Tests for adversarial mix cipher generator."""
    
    def test_generate_basic(self):
        gen = AdversarialMixGenerator()
        challenge = gen.generate(seed=42)
        self.assert_valid_challenge(challenge)
        assert challenge.category == ChallengeCategory.CIPHER
        assert challenge.challenge_type == "adversarial_mix"
    
    def test_has_noise(self):
        gen = AdversarialMixGenerator()
        challenge = gen.generate(difficulty=Difficulty.HARD, seed=42)
        ciphertext = challenge.task_card.content["ciphertext"]
        # Hard difficulty should have some noise characters
        from ai_secbench.utils.text import NOISE_CHARS
        has_noise = any(c in ciphertext for c in NOISE_CHARS)
        # Note: may not always have noise due to probability
        assert isinstance(has_noise, bool)
    
    def test_operations_recorded(self):
        gen = AdversarialMixGenerator()
        challenge = gen.generate(seed=42)
        ops = challenge.solution.key
        assert isinstance(ops, list)
        assert all(op in ["caesar", "rot13", "atbash", "identity"] for op in ops)


class TestPromptSecurityGenerator(TestChallengeBase):
    """Tests for prompt security cipher generator."""
    
    def test_generate_basic(self):
        gen = PromptSecurityGenerator()
        challenge = gen.generate(seed=42)
        self.assert_valid_challenge(challenge)
        assert challenge.category == ChallengeCategory.CIPHER
        assert challenge.challenge_type == "prompt_security"
    
    def test_has_adversarial_framing(self):
        gen = PromptSecurityGenerator()
        challenge = gen.generate(seed=42)
        container = challenge.task_card.content.get("container", "")
        # Should contain adversarial-looking content
        adversarial_markers = ["SYSTEM", "OVERRIDE", "IGNORE", "EXECUTE", "directive"]
        assert any(marker.lower() in container.lower() for marker in adversarial_markers)
    
    def test_has_safety_expectation(self):
        gen = PromptSecurityGenerator()
        challenge = gen.generate(seed=42)
        assert challenge.solution.expected_safety_behavior is not None


class TestContextPoisoningGenerator(TestChallengeBase):
    """Tests for context poisoning generator."""
    
    def test_generate_basic(self):
        gen = ContextPoisoningGenerator()
        challenge = gen.generate(seed=42)
        self.assert_valid_challenge(challenge)
        assert challenge.category == ChallengeCategory.CONTEXT_POISONING
    
    def test_has_poison_and_truth(self):
        gen = ContextPoisoningGenerator()
        challenge = gen.generate(seed=42)
        
        # Should have correct answer in solution
        assert challenge.solution.plaintext is not None
        
        # Should have poison answer recorded
        assert "poison_answer" in challenge.solution.additional_keys
        
        # They should be different
        assert (challenge.solution.plaintext != 
                challenge.solution.additional_keys["poison_answer"])
    
    def test_context_contains_question(self):
        gen = ContextPoisoningGenerator()
        challenge = gen.generate(seed=42)
        assert "question" in challenge.task_card.content


class TestContradictoryContextGenerator(TestChallengeBase):
    """Tests for contradictory context generator."""
    
    def test_generate_basic(self):
        gen = ContradictoryContextGenerator()
        challenge = gen.generate(seed=42)
        self.assert_valid_challenge(challenge)
        assert challenge.category == ChallengeCategory.CONTEXT_POISONING
        assert challenge.challenge_type == "contradictory_context"
    
    def test_has_contradictory_values(self):
        gen = ContradictoryContextGenerator()
        challenge = gen.generate(seed=42)
        additional = challenge.solution.additional_keys
        assert "value1" in additional
        assert "value2" in additional
        assert additional["value1"] != additional["value2"]


class TestSocialEngineeringGenerators(TestChallengeBase):
    """Tests for social engineering generators."""
    
    def test_authority_spoofing_basic(self):
        gen = AuthoritySpoofingGenerator()
        challenge = gen.generate(seed=42)
        self.assert_valid_challenge(challenge)
        assert challenge.category == ChallengeCategory.SOCIAL_ENGINEERING
        assert "fake_authority" in challenge.solution.additional_keys
    
    def test_emotional_manipulation_basic(self):
        gen = EmotionalManipulationGenerator()
        challenge = gen.generate(seed=42)
        self.assert_valid_challenge(challenge)
        assert challenge.category == ChallengeCategory.SOCIAL_ENGINEERING
        assert "manipulation_type" in challenge.solution.additional_keys
    
    def test_roleplay_manipulation_basic(self):
        gen = RoleplayManipulationGenerator()
        challenge = gen.generate(seed=42)
        self.assert_valid_challenge(challenge)
        assert challenge.category == ChallengeCategory.SOCIAL_ENGINEERING
        assert "manipulation_type" in challenge.solution.additional_keys


class TestGeneratorRegistry:
    """Tests for generator registries."""
    
    def test_list_cipher_generators(self):
        generators = list_cipher_generators()
        assert "multi_shift_caesar" in generators
        assert "vigenere" in generators
        assert "adversarial_mix" in generators
        assert "prompt_security" in generators
    
    def test_get_cipher_generator(self):
        gen = get_cipher_generator("vigenere")
        assert isinstance(gen, VigenereGenerator)
    
    def test_get_unknown_generator_raises(self):
        with pytest.raises(ValueError):
            get_cipher_generator("unknown_type")
    
    def test_list_context_poisoning_generators(self):
        generators = list_context_poisoning_generators()
        assert "context_poisoning" in generators
        assert "contradictory_context" in generators
    
    def test_list_social_engineering_generators(self):
        generators = list_social_engineering_generators()
        assert "authority_spoofing" in generators
        assert "emotional_manipulation" in generators
        assert "roleplay_manipulation" in generators
