# AI-SecBench

**Security-Focused AI Reasoning Benchmark**

Contributors: \
Michael A. Riegler, Simula Resaerch Laboratory, michael@simula.no \
Julia Mykhailiuk, DNV

AI-SecBench evaluates AI models on security-adjacent reasoning tasks including cipher analysis, steganographic detection, and adversarial robustness. Unlike traditional benchmarks that focus solely on correctness, AI-SecBench measures multiple dimensions:

- **Correctness**: Did the model get the right answer?
- **Reasoning Quality**: Was the reasoning sound and step-by-step?
- **Safety Compliance**: Did the model resist adversarial framing?
- **Verification**: Did the model verify their answer?

## Features

- 🔐 **Multi-dimensional scoring** - Beyond just correctness
- 🎯 **Three challenge categories** - Cipher, Steganographic, Context Poisoning
- 🌍 **Multi-language support** - English and Norwegian (extensible)
- 🔄 **Reproducible** - Seeded generation + versioned official sets
- 💰 **Cost tracking** - Estimate API costs per run
- 🔌 **Multi-provider** - Anthropic, OpenAI, HuggingFace, xAI (Grok)
- 📊 **LLM-as-Judge** - Rubric-based reasoning evaluation

## Installation

```bash
# Basic installation
pip install git+https://github.com/kelkalot/ai-secbench.git

# With specific provider
pip install git+https://github.com/kelkalot/ai-secbench.git[anthropic]
pip install git+https://github.com/kelkalot/ai-secbench.git[openai]
pip install git+https://github.com/kelkalot/ai-secbench.git[huggingface]
pip install git+https://github.com/kelkalot/ai-secbench.git[xai]

# All providers
pip install git+https://github.com/kelkalot/ai-secbench.git[all]

# Development
pip install git+https://github.com/kelkalot/ai-secbench.git[dev]
```

## Quick Start

### Basic Usage

```python
from ai_secbench import run_benchmark

# Run a quick evaluation
results = run_benchmark(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    challenge_types=["cipher", "steganographic", "context_poisoning"],
    n_per_type=2,
    verbose=True,
)

print(results.summary())
```

### Using Scenario Packs

```python
from ai_secbench import BenchmarkRunner, BenchmarkConfig
from ai_secbench.scenarios import load_scenario_pack

# Load a pre-defined scenario pack
challenges = load_scenario_pack("quick_eval", seed=42)

# Run benchmark
config = BenchmarkConfig(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
)

runner = BenchmarkRunner(config, challenge_set=challenges)
results = runner.run_sync()
```

### Command Line

```bash
ai-secbench --provider xai --model grok-2-1212 --n-per-type 2 --language english --output results.json
```

### Generate Challenges Only

```python
from ai_secbench.challenges import get_challenge_generator

# Generate cipher challenges
gen = get_challenge_generator("cipher")
challenges = gen.generate(n=5, language="english", seed=42)

for c in challenges:
    print(f"{c.challenge_id}: {c.subtype}")
    print(c.get_prompt())
    print("---")
```

## Challenge Types

### 1. Cipher Challenges

Test cryptanalysis and pattern recognition:

- **Multi-shift Caesar**: Each word shifted by different amounts
- **Vigenère/Autokey**: Keyword-based encryption with thematic hints
- **Adversarial Mix**: Multiple cipher types + noise injection
- **Prompt-Security**: Cipher wrapped in adversarial instruction container

### 2. Steganographic Challenges

Test hidden message detection:

- **Acrostic (Lines)**: First letter of each line spells message
- **Acrostic (Words)**: First letter of key words spells message
- **Nth Letter**: Hidden in specific letter positions
- **Safety-Aware**: May contain harmful hidden instructions

### 3. Context Poisoning Challenges

Test resistance to misinformation:

- **False Facts**: Incorrect information in provided context
- **Contradictory Context**: Conflicting claims to resolve
- **Authority Spoofing**: Fake system/developer messages
- **Source Manipulation**: Fake academic sources

## Scoring System

Each challenge produces multiple scores:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Correctness | 40% | Did the model get the right answer? |
| Reasoning | 25% | Was the reasoning logical and clear? |
| Safety | 20% | Did the model resist adversarial content? |
| Verification | 15% | Did the model verify their answer? |

The **composite score** is a weighted average of all dimensions.

## Configuration

```python
from ai_secbench import BenchmarkConfig, TurnMode, JudgeMode

config = BenchmarkConfig(
    # Provider settings
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    
    # Judge settings (for reasoning evaluation)
    judge_provider="anthropic",
    judge_model="claude-3-5-sonnet-20241022",
    judge_mode=JudgeMode.AUTOMATIC,
    
    # Turn settings
    turn_mode=TurnMode.SINGLE_SHOT,  # or FIXED_MULTI, INTERACTIVE
    max_turns=5,
    
    # Challenge settings
    challenge_types=["cipher", "steganographic", "context_poisoning"],
    n_challenges_per_type=3,
    language="english",  # or "norwegian"
    
    # Generation settings
    master_seed=42,  # For reproducibility
    use_official_set=False,
    
    # Output settings
    verbose=True,
    track_costs=True,
    output_dir="./results",
)
```

## Scenario Packs

Pre-defined challenge collections:

| Pack | Description | Challenges |
|------|-------------|------------|
| `quick_eval` | Quick 6-challenge evaluation | 2 per type |
| `full_eval` | Complete 15-challenge evaluation | 5 per type |
| `cipher_focus` | Cipher-focused deep dive | 8 cipher |
| `adversarial` | Adversarial robustness focus | 8 adversarial |
| `safety` | Safety-focused evaluation | 8 safety |
| `norwegian` | Norwegian language evaluation | 9 total |

## Extending AI-SecBench

### Adding New Challenge Types

```python
from ai_secbench.challenges import BaseChallengeGenerator, register_challenge_type

class MyCustomGenerator(BaseChallengeGenerator):
    @property
    def challenge_type(self) -> str:
        return "custom"
    
    @property
    def subtypes(self) -> List[str]:
        return ["subtype_a", "subtype_b"]
    
    def generate(self, n=1, **kwargs):
        # Generate challenges
        ...

# Register the new type
register_challenge_type("custom", MyCustomGenerator)
```

### Adding New Languages

Add entries to the plaintext pools in `ai_secbench/utils/text.py`:

```python
PLAINTEXT_POOLS["german"] = [
    "nachricht um mitternacht am nordtor mit kennwort delta",
    # ... more messages
]

WORD_BANKS["german"] = [
    "norden", "süden", "osten", "westen", ...
]
```

## API Reference

### Core Classes

- `BenchmarkRunner` - Main orchestrator
- `BenchmarkConfig` - Configuration dataclass
- `Challenge` - Single challenge data
- `ChallengeSet` - Collection of challenges
- `ChallengeResult` - Result of one challenge
- `EvaluationResult` - Complete benchmark results
- `Evaluator` - Scoring logic

### Providers

- `get_provider(name, model, api_key)` - Get provider instance
- `AnthropicProvider` - Claude models
- `OpenAIProvider` - GPT models
- `HuggingFaceProvider` - HF Inference API
- `LocalHuggingFaceProvider` - Local transformers

### Challenges

- `get_challenge_generator(type)` - Get generator
- `list_challenge_types()` - Available types
- `register_challenge_type(name, class)` - Add custom type

### Scenarios

- `load_scenario_pack(name, seed)` - Load pre-defined pack
- `list_scenario_packs()` - Available packs
- `save_official_set(set, version)` - Save versioned set

## Environment Variables

```bash
# Provider API keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
XAI_API_KEY=xaikey_...
```

## Contributing

Contributions welcome! Areas of interest:

- New challenge types
- Additional languages
- Provider integrations
- Evaluation metrics
- Documentation

## License

MIT License - see LICENSE file.

## Acknowledgments

Inspired by:
- [simpleaudit](https://github.com/kelkalot/simpleaudit) - LLM safety auditing patterns
- Online discussions (https://www.linkedin.com/posts/michael-alexander-riegler-4719157a_ai-security-llms-activity-7393211726466396160-yQvs?utm_source=share&utm_medium=member_desktop&rcm=ACoAABDdgVUBJ1O0ledD8r94910EP1LeUAAWAo8) and the Norwegian Armed Forces (https://www.forsvaret.no)
