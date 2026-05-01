"""Bootstrap helpers for the repo-local MeloTTS reference environment."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import textwrap
from pathlib import Path

MELO_GIT_URL = "https://github.com/myshell-ai/MeloTTS.git"
MELO_MODEL_REPOS = {
    "EN": "myshell-ai/MeloTTS-English",
    "EN_V2": "myshell-ai/MeloTTS-English-v2",
    "ES": "myshell-ai/MeloTTS-Spanish",
    "FR": "myshell-ai/MeloTTS-French",
    "JP": "myshell-ai/MeloTTS-Japanese",
    "KR": "myshell-ai/MeloTTS-Korean",
    "ZH": "myshell-ai/MeloTTS-Chinese",
}
PREFETCH_MODEL_IDS = {
    "en": "bert-base-uncased",
    "zh": "bert-base-multilingual-uncased",
}

PATCHED_SETUP_PY = textwrap.dedent(
    """\
    from setuptools import find_packages, setup


    setup(
        name="melotts",
        version="0.1.2",
        packages=find_packages(),
        include_package_data=True,
        package_data={"": ["*.txt", "cmudict_*"]},
        entry_points={
            "console_scripts": [
                "melotts = melo.main:main",
                "melo = melo.main:main",
                "melo-ui = melo.app:main",
            ],
        },
    )
    """
)

PATCHED_DOWNLOAD_UTILS = textwrap.dedent(
    """\
    from huggingface_hub import hf_hub_download
    import torch

    from . import utils


    REPO_IDS = {
        "EN": "myshell-ai/MeloTTS-English",
        "EN_V2": "myshell-ai/MeloTTS-English-v2",
        "ES": "myshell-ai/MeloTTS-Spanish",
        "FR": "myshell-ai/MeloTTS-French",
        "JP": "myshell-ai/MeloTTS-Japanese",
        "KR": "myshell-ai/MeloTTS-Korean",
        "ZH": "myshell-ai/MeloTTS-Chinese",
    }


    def _repo_id(locale):
        language = locale.split("-")[0].upper()
        assert language in REPO_IDS
        return REPO_IDS[language]


    def load_or_download_config(locale):
        config_path = hf_hub_download(repo_id=_repo_id(locale), filename="config.json")
        return utils.get_hparams_from_file(config_path)


    def load_or_download_model(locale, device):
        ckpt_path = hf_hub_download(repo_id=_repo_id(locale), filename="checkpoint.pth")
        return torch.load(ckpt_path, map_location=device)
    """
)

PATCHED_CLEANER = textwrap.dedent(
    """\
    import copy
    import importlib

    from . import cleaned_text_to_sequence


    LANGUAGE_MODULE_NAMES = {
        "ZH": ".chinese",
        "JP": ".japanese",
        "EN": ".english",
        "ZH_MIX_EN": ".chinese_mix",
        "KR": ".korean",
        "FR": ".french",
        "SP": ".spanish",
        "ES": ".spanish",
    }


    def _language_module(language):
        module_name = LANGUAGE_MODULE_NAMES[language]
        return importlib.import_module(module_name, __package__)


    def clean_text(text, language):
        language_module = _language_module(language)
        norm_text = language_module.text_normalize(text)
        phones, tones, word2ph = language_module.g2p(norm_text)
        return norm_text, phones, tones, word2ph


    def clean_text_bert(text, language, device=None):
        language_module = _language_module(language)
        norm_text = language_module.text_normalize(text)
        phones, tones, word2ph = language_module.g2p(norm_text)

        word2ph_bak = copy.deepcopy(word2ph)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
        bert = language_module.get_bert_feature(norm_text, word2ph, device=device)

        return norm_text, phones, tones, word2ph_bak, bert


    def text_to_sequence(text, language):
        norm_text, phones, tones, word2ph = clean_text(text, language)
        return cleaned_text_to_sequence(phones, tones, language)
    """
)

PATCHED_TEXT_INIT = textwrap.dedent(
    """\
    import importlib

    from .symbols import *


    _symbol_to_id = {s: i for i, s in enumerate(symbols)}


    def cleaned_text_to_sequence(cleaned_text, tones, language, symbol_to_id=None):
        symbol_to_id_map = symbol_to_id if symbol_to_id else _symbol_to_id
        phones = [symbol_to_id_map[symbol] for symbol in cleaned_text]
        tone_start = language_tone_start_map[language]
        tones = [i + tone_start for i in tones]
        lang_id = language_id_map[language]
        lang_ids = [lang_id for _ in phones]
        return phones, tones, lang_ids


    BERT_MODULES = {
        "ZH": ".chinese_bert",
        "JP": ".japanese_bert",
        "EN": ".english_bert",
        "ZH_MIX_EN": ".chinese_mix",
        "KR": ".korean",
        "FR": ".french_bert",
        "SP": ".spanish_bert",
        "ES": ".spanish_bert",
    }


    def get_bert(norm_text, word2ph, language, device):
        module = importlib.import_module(BERT_MODULES[language], __package__)
        return module.get_bert_feature(norm_text, word2ph, device)
    """
)

CPU_SAFE_BERT_BLOCK = (
    "    if not device:\n"
    "        if torch.cuda.is_available():\n"
    '            device = "cuda"\n'
    "        else:\n"
    '            device = "cpu"\n'
)

UPSTREAM_BERT_BLOCK = (
    "    if (\n"
    '        sys.platform == "darwin"\n'
    "        and torch.backends.mps.is_available()\n"
    '        and device == "cpu"\n'
    "    ):\n"
    '        device = "mps"\n'
    "    if not device:\n"
    '        device = "cuda"\n'
)

UPSTREAM_AUTO_DEVICE_BLOCK = (
    "        if device == 'auto':\n"
    "            device = 'cpu'\n"
    "            if torch.cuda.is_available(): device = 'cuda'\n"
    "            if torch.backends.mps.is_available(): device = 'mps'\n"
)

PATCHED_AUTO_DEVICE_BLOCK = (
    "        if device == 'auto':\n"
    "            device = 'cpu'\n"
    "            if torch.cuda.is_available():\n"
    "                device = 'cuda'\n"
)

UPSTREAM_ENGLISH_IMPORT = "from .japanese import distribute_phone\n"
UPSTREAM_ENGLISH_TOKENIZER = textwrap.dedent(
    """\
    model_id = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    """
)

PATCHED_ENGLISH_TOKENIZER = textwrap.dedent(
    """\
    model_id = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_id)


    def distribute_phone(n_phone, n_word):
        phones_per_word = [0] * n_word
        for _ in range(n_phone):
            min_tasks = min(phones_per_word)
            min_index = phones_per_word.index(min_tasks)
            phones_per_word[min_index] += 1
        return phones_per_word
    """
)


def runtime_requirements() -> tuple[str, ...]:
    """Return the tested runtime requirements for the Melo reference env."""
    return (
        "aiohttp>=3.11,<4",
        "cn2an==0.5.22",
        "g2p_en==2.1.0",
        "inflect==7.0.0",
        "jieba==0.42.1",
        "librosa==0.10.2.post1",
        "num2words==0.5.12",
        "numpy<2",
        "pypinyin==0.50.0",
        "setuptools<81",
        "soundfile",
        "torch==2.11.0",
        "torchaudio==2.11.0",
        "transformers==4.27.4",
        "txtsplit",
        "tqdm",
        "unidecode==1.3.7",
    )


def _replace_or_fail(text: str, old: str, new: str, *, file_path: Path) -> str:
    if new and new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Could not find the expected upstream block in {file_path}.")
    return text.replace(old, new)


def write_runtime_requirements(path: Path) -> None:
    """Write the tested Melo runtime requirements file."""
    content = "# Tested Apple Silicon Melo runtime for zh/en sidecar.\n"
    content += "\n".join(runtime_requirements()) + "\n"
    path.write_text(content)


def ensure_source_checkout(*, source_dir: Path, git_ref: str) -> None:
    """Create a fresh Melo source checkout at the requested ref."""
    if source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", MELO_GIT_URL, str(source_dir)], check=True)
    subprocess.run(["git", "checkout", git_ref], cwd=source_dir, check=True)


def patch_reference_source(source_dir: Path) -> None:
    """Apply the repo-owned zh/en Apple Silicon patch layer to Melo."""
    (source_dir / "setup.py").write_text(PATCHED_SETUP_PY)
    (source_dir / "melo" / "download_utils.py").write_text(PATCHED_DOWNLOAD_UTILS)
    (source_dir / "melo" / "text" / "cleaner.py").write_text(PATCHED_CLEANER)
    (source_dir / "melo" / "text" / "__init__.py").write_text(PATCHED_TEXT_INIT)

    api_path = source_dir / "melo" / "api.py"
    api_text = api_path.read_text()
    api_path.write_text(
        _replace_or_fail(
            api_text,
            UPSTREAM_AUTO_DEVICE_BLOCK,
            PATCHED_AUTO_DEVICE_BLOCK,
            file_path=api_path,
        )
    )

    english_path = source_dir / "melo" / "text" / "english.py"
    english_text = english_path.read_text()
    english_text = _replace_or_fail(
        english_text,
        UPSTREAM_ENGLISH_IMPORT,
        "",
        file_path=english_path,
    )
    english_text = _replace_or_fail(
        english_text,
        UPSTREAM_ENGLISH_TOKENIZER,
        PATCHED_ENGLISH_TOKENIZER,
        file_path=english_path,
    )
    english_path.write_text(english_text)

    for name in ("chinese_bert.py", "english_bert.py"):
        bert_path = source_dir / "melo" / "text" / name
        bert_text = bert_path.read_text()
        bert_path.write_text(
            _replace_or_fail(
                bert_text,
                UPSTREAM_BERT_BLOCK,
                CPU_SAFE_BERT_BLOCK,
                file_path=bert_path,
            )
        )


def _normalize_prefetch_language(value: str) -> str:
    normalized = value.strip().lower()
    if normalized.startswith("zh"):
        return "zh"
    if normalized.startswith("en"):
        return "en"
    raise ValueError(f"Unsupported prefetch language `{value}`.")


def prefetch_runtime_assets(languages: list[str]) -> None:
    """Prefetch zh/en assets so the first sidecar run is not download-bound."""
    import nltk
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    normalized_languages = sorted({_normalize_prefetch_language(value) for value in languages})
    nltk.download("averaged_perceptron_tagger", quiet=False)
    nltk.download("averaged_perceptron_tagger_eng", quiet=False)
    nltk.download("cmudict", quiet=False)

    for language in normalized_languages:
        if language == "zh":
            hf_hub_download(repo_id=MELO_MODEL_REPOS["ZH"], filename="config.json")
            hf_hub_download(repo_id=MELO_MODEL_REPOS["ZH"], filename="checkpoint.pth")
        if language == "en":
            hf_hub_download(repo_id=MELO_MODEL_REPOS["EN"], filename="config.json")
            hf_hub_download(repo_id=MELO_MODEL_REPOS["EN"], filename="checkpoint.pth")

        model_id = PREFETCH_MODEL_IDS[language]
        AutoTokenizer.from_pretrained(model_id)
        AutoModelForMaskedLM.from_pretrained(model_id)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Manage the repo-local Melo reference source.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-source")
    prepare.add_argument("--source-dir", required=True)
    prepare.add_argument("--git-ref", required=True)

    requirements = subparsers.add_parser("write-runtime-requirements")
    requirements.add_argument("--output", required=True)

    prefetch = subparsers.add_parser("prefetch")
    prefetch.add_argument("--language", action="append", required=True)

    return parser


def main() -> int:
    """CLI entrypoint for the Melo reference helper."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare-source":
        source_dir = Path(args.source_dir)
        ensure_source_checkout(source_dir=source_dir, git_ref=args.git_ref)
        patch_reference_source(source_dir)
        return 0

    if args.command == "write-runtime-requirements":
        write_runtime_requirements(Path(args.output))
        return 0

    if args.command == "prefetch":
        prefetch_runtime_assets(list(args.language))
        return 0

    parser.error(f"Unknown command `{args.command}`.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
