import os
import sys
import shutil
from pathlib import Path
from dotenv import load_dotenv

# --- Dependency Checks ---
def _check_install(package_name: str, import_name: str = None):
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing missing dependency: {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Installed {package_name}.")

import subprocess
try:
    import pypdf
except ImportError:
    _check_install("pypdf")
    import pypdf

# -------------------------

def setup_environment():
    """
    Load environment variables from .env file and check for required keys.
    """
    # Load from project root .env
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found in environment variables.")
        print(f"Checked path: {env_path.absolute()}")
        print("Please ensure your .env file exists and contains OPENAI_API_KEY.")

def print_section(title: str):
    """
    Print a section header to the console.
    """
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")

from rag_lib.core.domain import Segment, SegmentType

def load_text_segment(file_path: Path, format_type: str = "text") -> Segment:
    if not file_path.exists():
        print(f"Warning: File {file_path} not found.")
        return None
    print(f"Loading {file_path.name}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return Segment(
            content=content,
            type=SegmentType.TEXT,
            original_format=format_type,
            path=[file_path.name],
            metadata={"source_file": file_path.name}
        )
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return None

def load_pdf_text(file_path: Path) -> Segment:
    if not file_path.exists():
        print(f"Warning: File {file_path} not found.")
        return None
    
    print(f"Loading PDF text from {file_path.name}...")
    try:
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n\n"
        
        return Segment(
            content=text,
            type=SegmentType.TEXT,
            original_format="pdf",
            path=[file_path.name],
            metadata={
                "source_file": file_path.name, 
                "page_count": len(reader.pages)
            }
        )
    except Exception as e:
        print(f"Error loading PDF {file_path.name}: {e}")
        return None
