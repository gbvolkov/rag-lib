import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from rag_lib.loaders.miner_u import MinerULoader


def test_miner_u_missing_dependency():
    with patch("rag_lib.loaders.miner_u.importlib.util.find_spec", return_value=None), patch.object(
        MinerULoader, "_resolve_cli_command", return_value=None
    ):
        with pytest.raises(ImportError) as excinfo:
            MinerULoader(file_path="dummy.pdf")
        assert "MinerU is not available" in str(excinfo.value)


def test_miner_u_load_returns_markdown_document(tmp_path: Path):
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    with patch.object(MinerULoader, "_ensure_available", return_value=None):
        loader = MinerULoader(file_path=str(pdf_path), method="auto")

    with patch.object(loader, "_run_mineru_cli", return_value=["mineru", "-p"]), patch.object(
        loader, "_read_markdown_output", return_value="# Heading\n\nBody"
    ):
        docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "# Heading\n\nBody"
    assert docs[0].metadata["source"] == str(pdf_path)
    assert docs[0].metadata["parser"] == "MinerU"
    assert docs[0].metadata["output_format"] == "markdown"
    assert docs[0].metadata["mineru_method"] == "auto"


def test_run_mineru_cli_fallback_to_second_command(tmp_path: Path):
    with patch.object(MinerULoader, "_ensure_available", return_value=None):
        loader = MinerULoader(file_path=str(tmp_path / "dummy.pdf"))

    with patch.object(loader, "_candidate_commands", return_value=[["mineru"], ["python", "-m", "mineru"]]), patch(
        "rag_lib.loaders.miner_u.subprocess.run",
        side_effect=[
            subprocess.CalledProcessError(returncode=1, cmd=["mineru"], stderr="failed"),
            subprocess.CompletedProcess(args=["python", "-m", "mineru"], returncode=0, stdout="", stderr=""),
        ],
    ):
        used = loader._run_mineru_cli(str(tmp_path))

    assert list(used) == ["python", "-m", "mineru"]


def test_run_mineru_cli_treats_traceback_output_as_failure(tmp_path: Path):
    with patch.object(MinerULoader, "_ensure_available", return_value=None):
        loader = MinerULoader(file_path=str(tmp_path / "dummy.pdf"))

    with patch.object(loader, "_candidate_commands", return_value=[["mineru"]]), patch(
        "rag_lib.loaders.miner_u.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["mineru"],
            returncode=0,
            stdout="",
            stderr="Traceback (most recent call last)\nrequests.exceptions.ProxyError",
        ),
    ):
        with pytest.raises(RuntimeError) as excinfo:
            loader._run_mineru_cli(str(tmp_path))
    assert "Failed to parse PDF with MinerU" in str(excinfo.value)


def test_read_markdown_output_picks_largest_md(tmp_path: Path):
    small = tmp_path / "small.md"
    big = tmp_path / "big.md"
    small.write_text("small", encoding="utf-8")
    big.write_text("## Title\n" + ("content " * 20), encoding="utf-8")

    with patch.object(MinerULoader, "_ensure_available", return_value=None):
        loader = MinerULoader(file_path=str(tmp_path / "dummy.pdf"))

    content = loader._read_markdown_output(str(tmp_path))
    assert "## Title" in content


def test_read_markdown_output_includes_proxy_hint_when_no_md(tmp_path: Path):
    with patch.object(MinerULoader, "_ensure_available", return_value=None):
        loader = MinerULoader(file_path=str(tmp_path / "dummy.pdf"))
    loader._last_cli_stderr = "Traceback (most recent call last)"
    with patch.dict(
        "os.environ",
        {"HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9"},
        clear=False,
    ):
        with pytest.raises(RuntimeError) as excinfo:
            loader._read_markdown_output(str(tmp_path))
    msg = str(excinfo.value)
    assert "produced no markdown output" in msg
    assert "Detected proxy environment variables" in msg


def test_resolve_cli_command_finds_interpreter_script(tmp_path: Path):
    scripts_dir = tmp_path / ("Scripts" if os.name == "nt" else "bin")
    scripts_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if os.name == "nt" else ""
    mineru_path = scripts_dir / f"mineru{suffix}"
    mineru_path.write_text("stub", encoding="utf-8")

    fake_python = scripts_dir / ("python.exe" if os.name == "nt" else "python")
    fake_python.write_text("stub", encoding="utf-8")

    with patch.object(MinerULoader, "_ensure_available", return_value=None):
        loader = MinerULoader(file_path=str(tmp_path / "dummy.pdf"))

    with patch("rag_lib.loaders.miner_u.shutil.which", return_value=None), patch(
        "rag_lib.loaders.miner_u.sys.executable", str(fake_python)
    ):
        resolved = loader._resolve_cli_command("mineru")

    assert resolved == str(mineru_path)


def test_ensure_available_handles_missing_mineru_main_module():
    def _fake_find_spec(name: str):
        if name == "mineru.__main__":
            raise ModuleNotFoundError("No module named 'mineru'")
        if name == "mineru":
            class _Spec:  # lightweight sentinel spec object
                pass
            return _Spec()
        return None

    with patch("rag_lib.loaders.miner_u.importlib.util.find_spec", side_effect=_fake_find_spec), patch.object(
        MinerULoader, "_resolve_cli_command", return_value=None
    ):
        loader = MinerULoader(file_path="dummy.pdf")
        assert loader.file_path == "dummy.pdf"


def test_candidate_commands_include_cli_options(tmp_path: Path):
    with patch.object(MinerULoader, "_ensure_available", return_value=None), patch.object(
        MinerULoader, "_resolve_cli_command", return_value="mineru"
    ), patch.object(MinerULoader, "_has_module_spec", return_value=False):
        loader = MinerULoader(
            file_path=str(tmp_path / "dummy.pdf"),
            method="txt",
            backend="pipeline",
            lang="cyrillic",
            server_url="http://127.0.0.1:30000",
            start_page=0,
            end_page=5,
            parse_formula=False,
            parse_table=True,
            device="cpu",
            vram=1,
            source="huggingface",
        )
        commands = loader._candidate_commands(str(tmp_path / "out"))

    command = commands[0]
    assert "-b" in command and "pipeline" in command
    assert "--source" in command and "huggingface" in command
    assert "-s" in command and "0" in command
    assert "-e" in command and "5" in command
