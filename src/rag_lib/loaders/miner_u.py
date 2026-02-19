from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from rag_lib.core.domain import Document
from rag_lib.core.logger import logger


class MinerULoader:
    """
    Loader using MinerU for layout-aware PDF parsing.
    Returns one markdown Document per input PDF.
    """

    def __init__(
        self,
        file_path: str,
        parse_mode: str = "auto",
        backend: Optional[str] = None,
        lang: Optional[str] = None,
        server_url: Optional[str] = None,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        parse_formula: Optional[bool] = None,
        parse_table: Optional[bool] = None,
        device: Optional[str] = None,
        vram: Optional[int] = None,
        source: Optional[str] = None,
        timeout_seconds: int = 600,
        keep_temp_artifacts: bool = False,
    ):
        self.file_path = file_path
        self.parse_mode = parse_mode
        self.backend = backend
        self.lang = lang
        self.server_url = server_url
        self.start_page = start_page
        self.end_page = end_page
        self.parse_formula = parse_formula
        self.parse_table = parse_table
        self.device = device
        self.vram = vram
        self.source = source
        self.timeout_seconds = timeout_seconds
        self.keep_temp_artifacts = keep_temp_artifacts
        self._last_cli_stdout = ""
        self._last_cli_stderr = ""

        if self.parse_mode not in {"auto", "txt", "ocr"}:
            raise ValueError("parse_mode must be one of: auto, txt, ocr")
        if self.backend is not None and self.backend not in {
            "pipeline",
            "hybrid-auto-engine",
            "hybrid-http-client",
            "vlm-auto-engine",
            "vlm-http-client",
        }:
            raise ValueError("invalid backend")
        if self.source is not None and self.source not in {"huggingface", "modelscope", "local"}:
            raise ValueError("source must be one of: huggingface, modelscope, local")
        if self.start_page is not None and self.start_page < 0:
            raise ValueError("start_page must be >= 0")
        if self.end_page is not None and self.end_page < 0:
            raise ValueError("end_page must be >= 0")
        if self.start_page is not None and self.end_page is not None and self.end_page < self.start_page:
            raise ValueError("end_page must be >= start_page")

        self._ensure_available()

    def load(self) -> List[Document]:
        """
        Run MinerU conversion and return markdown as one Document.
        """
        logger.info(f"Loading PDF with MinerU: {self.file_path}")

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        output_dir = tempfile.mkdtemp(prefix="mineru_")
        try:
            used_command = self._run_mineru_cli(output_dir)
            markdown = self._read_markdown_output(output_dir)
        finally:
            if self.keep_temp_artifacts:
                logger.debug(f"Keeping MinerU temp artifacts at: {output_dir}")
            else:
                # MinerU may keep files open briefly on Windows; ignore cleanup errors.
                shutil.rmtree(output_dir, ignore_errors=True)

        metadata = {
            "source": self.file_path,
            "parser": "MinerU",
            "output_format": "markdown",
            "mineru_parse_mode": self.parse_mode,
            "mineru_command": " ".join(used_command),
        }
        if self.backend:
            metadata["mineru_backend"] = self.backend
        if self.lang:
            metadata["lang"] = self.lang
        if self.source:
            metadata["mineru_source"] = self.source
        if self.start_page is not None:
            metadata["start_page"] = self.start_page
        if self.end_page is not None:
            metadata["end_page"] = self.end_page

        return [Document(page_content=markdown, metadata=metadata)]

    def _ensure_available(self) -> None:
        has_mineru_module = self._has_module_spec("mineru")
        has_mineru_main = self._has_module_spec("mineru.__main__")
        has_mineru_cli = self._resolve_cli_command("mineru") is not None

        if has_mineru_cli or has_mineru_main or has_mineru_module:
            return

        raise ImportError(
            "MinerU is not available. Install with `pip install mineru[all]` or "
            "`pip install rag-lib[miner_u]`."
        )

    def _has_module_spec(self, module_name: str) -> bool:
        """
        Safe wrapper around find_spec for dotted module names.
        For absent parent packages (e.g., mineru.__main__), find_spec can raise.
        """
        try:
            return importlib.util.find_spec(module_name) is not None
        except ModuleNotFoundError:
            return False

    def _resolve_cli_command(self, command: str) -> Optional[str]:
        """
        Resolve CLI command path from the current interpreter's scripts
        directory first, then PATH (useful for IDE runs without activation).
        """
        scripts_dir = Path(sys.executable).resolve().parent
        names = [command]
        if os.name == "nt" and not command.lower().endswith(".exe"):
            names.append(f"{command}.exe")

        for name in names:
            candidate = scripts_dir / name
            if candidate.is_file():
                return str(candidate)

        on_path = shutil.which(command)
        if on_path:
            return on_path

        return None

    def _candidate_commands(self, output_dir: str) -> List[List[str]]:
        base_args = ["-p", self.file_path, "-o", output_dir, "-m", self.parse_mode]
        if self.backend:
            base_args.extend(["-b", self.backend])
        if self.lang:
            base_args.extend(["-l", self.lang])
        if self.server_url:
            base_args.extend(["-u", self.server_url])
        if self.start_page is not None:
            base_args.extend(["-s", str(self.start_page)])
        if self.end_page is not None:
            base_args.extend(["-e", str(self.end_page)])
        if self.parse_formula is not None:
            base_args.extend(["-f", str(self.parse_formula).lower()])
        if self.parse_table is not None:
            base_args.extend(["-t", str(self.parse_table).lower()])
        if self.device:
            base_args.extend(["-d", self.device])
        if self.vram is not None:
            base_args.extend(["--vram", str(self.vram)])
        if self.source:
            base_args.extend(["--source", self.source])

        candidates: List[List[str]] = []
        seen: set[str] = set()

        mineru_cmd = self._resolve_cli_command("mineru")
        if mineru_cmd:
            normalized = str(Path(mineru_cmd).resolve()).lower()
            if normalized not in seen:
                candidates.append([mineru_cmd, *base_args])
                seen.add(normalized)

        if self._has_module_spec("mineru.__main__"):
            candidates.append([sys.executable, "-m", "mineru", *base_args])

        return candidates

    def _run_mineru_cli(self, output_dir: str) -> Sequence[str]:
        candidates = self._candidate_commands(output_dir)
        if not candidates:
            raise RuntimeError(
                "MinerU CLI is not available. Checked PATH and interpreter scripts directory."
            )

        errors: List[Tuple[Sequence[str], str]] = []
        timeout_seen = False
        for command in candidates:
            logger.debug(f"Trying MinerU command: {' '.join(command)}")
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=True,
                    timeout=self.timeout_seconds,
                )
                self._last_cli_stdout = self._coerce_stream_text(completed.stdout)
                self._last_cli_stderr = self._coerce_stream_text(completed.stderr)

                if self._last_cli_stdout:
                    logger.debug(self._last_cli_stdout.strip())
                if self._last_cli_stderr:
                    logger.debug(self._last_cli_stderr.strip())

                if self._looks_like_cli_failure(self._last_cli_stdout, self._last_cli_stderr):
                    errors.append((command, self._summarize_cli_output(self._last_cli_stdout, self._last_cli_stderr)))
                    continue

                return command
            except subprocess.CalledProcessError as e:
                self._last_cli_stdout = self._coerce_stream_text(e.stdout)
                self._last_cli_stderr = self._coerce_stream_text(e.stderr)
                errors.append((command, self._summarize_cli_output(self._last_cli_stdout, self._last_cli_stderr)))
            except subprocess.TimeoutExpired as e:
                timeout_seen = True
                self._last_cli_stdout = self._coerce_stream_text(e.stdout)
                self._last_cli_stderr = self._coerce_stream_text(e.stderr)
                details = self._summarize_cli_output(self._last_cli_stdout, self._last_cli_stderr)
                errors.append(
                    (
                        command,
                        f"timeout after {self.timeout_seconds}s: {e}\nCLI output tail:\n{details}",
                    )
                )

        error_lines = [f"{' '.join(cmd)} -> {err}" for cmd, err in errors]
        message = "Failed to parse PDF with MinerU. Tried commands:\n" + "\n".join(error_lines)
        if timeout_seen:
            message += (
                "\nHint: MinerU may be slow on large PDFs. "
                "Try setting start_page/end_page to a smaller range "
                "(for example start_page=0, end_page=4) or increase timeout_seconds."
            )
        proxy_hint = self._proxy_hint()
        if proxy_hint:
            message += "\n" + proxy_hint
        raise RuntimeError(message)

    def _read_markdown_output(self, output_dir: str) -> str:
        md_files = [p for p in Path(output_dir).rglob("*.md") if p.is_file()]
        if not md_files:
            details = self._summarize_cli_output(self._last_cli_stdout, self._last_cli_stderr)
            proxy_hint = self._proxy_hint()
            raise RuntimeError(
                "MinerU finished but produced no markdown output.\n"
                f"CLI output tail:\n{details}\n"
                f"{proxy_hint}"
            )

        # Prefer the richest markdown file when multiple files are present.
        best_file = max(md_files, key=lambda p: p.stat().st_size)
        content = best_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            raise RuntimeError(f"MinerU markdown output is empty: {best_file}")

        return content

    def _looks_like_cli_failure(self, stdout: str, stderr: str) -> bool:
        combined = f"{stdout}\n{stderr}".lower()
        failure_markers = [
            "traceback (most recent call last)",
            "requests.exceptions.proxyerror",
            "urllib3.exceptions.proxyerror",
            "maxretryerror(",
            "modulenotfounderror:",
        ]
        return any(marker in combined for marker in failure_markers)

    def _summarize_cli_output(self, stdout: str, stderr: str, max_lines: int = 30) -> str:
        text = (stderr or "").strip()
        if not text:
            text = (stdout or "").strip()
        if not text:
            return "(no cli output captured)"

        lines = [line for line in text.splitlines() if line.strip()]
        tail = "\n".join(lines[-max_lines:])
        # Keep exception messages readable without flooding logs.
        return tail[-5000:]

    def _proxy_hint(self) -> str:
        proxy_keys = [k for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY") if os.environ.get(k)]
        if not proxy_keys:
            return ""
        return (
            "Detected proxy environment variables: "
            + ", ".join(proxy_keys)
            + ". If proxy is unreachable, MinerU model download will fail."
        )

    def _coerce_stream_text(self, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)
