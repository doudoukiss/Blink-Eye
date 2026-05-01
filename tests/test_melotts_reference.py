from pathlib import Path

from local_tts_servers import melotts_reference


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _fake_upstream_tree(root: Path) -> None:
    _write(root / "setup.py", "from pip.req import parse_requirements\n")
    _write(
        root / "melo" / "download_utils.py",
        "from cached_path import cached_path\nDOWNLOAD_CKPT_URLS = {}\n",
    )
    _write(
        root / "melo" / "text" / "cleaner.py",
        "from . import chinese, japanese, english, chinese_mix, korean, french, spanish\n",
    )
    _write(
        root / "melo" / "text" / "__init__.py",
        "from .chinese_bert import get_bert_feature as zh_bert\n"
        "from .english_bert import get_bert_feature as en_bert\n"
        "from .japanese_bert import get_bert_feature as jp_bert\n"
        "from .chinese_mix import get_bert_feature as zh_mix_en_bert\n"
        "from .spanish_bert import get_bert_feature as sp_bert\n"
        "from .french_bert import get_bert_feature as fr_bert\n"
        "from .korean import get_bert_feature as kr_bert\n",
    )
    _write(
        root / "melo" / "api.py",
        melotts_reference.UPSTREAM_AUTO_DEVICE_BLOCK,
    )
    _write(
        root / "melo" / "text" / "english.py",
        melotts_reference.UPSTREAM_ENGLISH_IMPORT + melotts_reference.UPSTREAM_ENGLISH_TOKENIZER,
    )
    _write(
        root / "melo" / "text" / "chinese_bert.py",
        "import sys\nimport torch\n" + melotts_reference.UPSTREAM_BERT_BLOCK,
    )
    _write(
        root / "melo" / "text" / "english_bert.py",
        "import sys\nimport torch\n" + melotts_reference.UPSTREAM_BERT_BLOCK,
    )


def test_patch_reference_source_applies_repo_owned_zh_en_patch(tmp_path):
    source_dir = tmp_path / "MeloTTS"
    _fake_upstream_tree(source_dir)

    melotts_reference.patch_reference_source(source_dir)

    assert "huggingface_hub" in (source_dir / "melo" / "download_utils.py").read_text()
    assert "importlib.import_module" in (source_dir / "melo" / "text" / "cleaner.py").read_text()
    assert "BERT_MODULES" in (source_dir / "melo" / "text" / "__init__.py").read_text()
    assert (
        "from .japanese import distribute_phone"
        not in (source_dir / "melo" / "text" / "english.py").read_text()
    )
    assert 'device = "mps"' not in (source_dir / "melo" / "text" / "chinese_bert.py").read_text()
    assert (
        "torch.backends.mps.is_available"
        not in (source_dir / "melo" / "text" / "english_bert.py").read_text()
    )
    assert "torch.backends.mps.is_available" not in (source_dir / "melo" / "api.py").read_text()


def test_runtime_requirements_keep_tested_melo_stack():
    requirements = melotts_reference.runtime_requirements()

    assert "librosa==0.10.2.post1" in requirements
    assert "torch==2.11.0" in requirements
    assert "torchaudio==2.11.0" in requirements
    assert "transformers==4.27.4" in requirements
    assert "numpy<2" in requirements
    assert "setuptools<81" in requirements
    assert "librosa==0.9.1" not in requirements


def test_prefetch_language_normalization_is_zh_en_only():
    assert melotts_reference._normalize_prefetch_language("zh-CN") == "zh"
    assert melotts_reference._normalize_prefetch_language("en-US") == "en"
