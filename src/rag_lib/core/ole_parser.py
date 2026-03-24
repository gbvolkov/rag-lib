from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import olefile


OLE_SIGNATURE = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"
WORD_DOCUMENT_STREAM = "WordDocument"
TABLE_STREAM_0 = "0Table"
TABLE_STREAM_1 = "1Table"

FIB_BASE_SIZE = 32
FIB97_CCP_TEXT_OFFSET = 12
FIB97_FC_CLX_PAIR_INDEX = 33

FIB_ENCRYPTED_MASK = 0x0100
FIB_TABLE_STREAM_MASK = 0x0200

CLX_PRL_MARKER = 0x01
CLX_PCDT_MARKER = 0x02

FIELD_BEGIN = "\x13"
FIELD_SEPARATOR = "\x14"
FIELD_END = "\x15"

COMPRESSED_ANSI_EXCEPTIONS = {
    0x82: 0x201A,
    0x83: 0x0192,
    0x84: 0x201E,
    0x85: 0x2026,
    0x86: 0x2020,
    0x87: 0x2021,
    0x88: 0x02C6,
    0x89: 0x2030,
    0x8A: 0x0160,
    0x8B: 0x2039,
    0x8C: 0x0152,
    0x91: 0x2018,
    0x92: 0x2019,
    0x93: 0x201C,
    0x94: 0x201D,
    0x95: 0x2022,
    0x96: 0x2013,
    0x97: 0x2014,
    0x98: 0x02DC,
    0x99: 0x2122,
    0x9A: 0x0161,
    0x9B: 0x203A,
    0x9C: 0x0153,
    0x9F: 0x0178,
}


class OleParserError(RuntimeError):
    """Base exception for strict Word binary parsing errors."""


class NotOleWordDocumentError(OleParserError):
    """Raised when a file is not a legacy Word OLE document."""


class EncryptedWordDocumentError(OleParserError):
    """Raised when the document is encrypted or obfuscated."""


class InvalidWordDocumentError(OleParserError):
    """Raised when required Word binary structures are missing or malformed."""


@dataclass(frozen=True)
class FibInfo:
    n_fib: int
    flags: int
    table_stream_name: str
    ccp_text: int
    fc_clx: int
    lcb_clx: int


@dataclass(frozen=True)
class TextPiece:
    cp_start: int
    cp_end: int
    file_offset: int
    is_compressed: bool


def extract_doc_text(file_path: str | Path) -> str:
    """
    Extract the main-document text from a legacy Word `.doc` file.

    This follows the Word binary piece-table path:
    WordDocument -> FIB -> CLX -> PlcPcd -> text pieces.
    """
    path = Path(file_path)
    _assert_ole_signature(path)

    ole = olefile.OleFileIO(str(path))
    try:
        word_document = _read_required_stream(ole, WORD_DOCUMENT_STREAM)
        fib = _read_fib(word_document)
        table_stream = _read_required_stream(ole, fib.table_stream_name)
        pieces = _read_text_pieces(table_stream, fib)
        text = _decode_main_document_text(word_document, pieces, fib.ccp_text)
    finally:
        ole.close()

    return text.replace("\r\n", "\n").replace("\r", "\n")


def cleanup_extracted_text(text: str) -> str:
    """
    Apply explicit Word-specific cleanup to extracted text.

    This is optional post-processing for readability; extraction itself stays raw.
    """
    cleaned = _keep_field_results_only(text)
    cleaned = cleaned.replace("\x07", "\t")
    cleaned = cleaned.replace("\x01", "")
    return cleaned


def _assert_ole_signature(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    if not path.is_file():
        raise IsADirectoryError(path)

    with path.open("rb") as handle:
        signature = handle.read(len(OLE_SIGNATURE))

    if signature != OLE_SIGNATURE:
        raise NotOleWordDocumentError(f"{path} is not an OLE compound file.")


def _read_required_stream(ole: olefile.OleFileIO, stream_name: str) -> bytes:
    if not ole.exists(stream_name):
        raise InvalidWordDocumentError(f"Missing required OLE stream: {stream_name}")
    return ole.openstream(stream_name).read()


def _read_fib(word_document: bytes) -> FibInfo:
    if len(word_document) < FIB_BASE_SIZE:
        raise InvalidWordDocumentError("WordDocument stream is too small to contain a FIB.")

    w_ident = _u16(word_document, 0)
    if w_ident != 0xA5EC:
        raise NotOleWordDocumentError(f"Unexpected WordDocument magic: 0x{w_ident:04X}")

    n_fib = _u16(word_document, 2)
    flags = _u16(word_document, 0x0A)

    if flags & FIB_ENCRYPTED_MASK:
        raise EncryptedWordDocumentError("Encrypted legacy Word documents are not supported.")
    if flags & 0x8000:
        raise EncryptedWordDocumentError("Obfuscated legacy Word documents are not supported.")

    table_stream_name = TABLE_STREAM_1 if flags & FIB_TABLE_STREAM_MASK else TABLE_STREAM_0

    pos = FIB_BASE_SIZE
    csw = _u16(word_document, pos)
    pos += 2 + (csw * 2)

    cslw = _u16(word_document, pos)
    pos += 2
    fib_lw_size = cslw * 4
    fib_lw = _slice_exact(word_document, pos, fib_lw_size, "FibRgLw")
    pos += fib_lw_size

    if len(fib_lw) < FIB97_CCP_TEXT_OFFSET + 4:
        raise InvalidWordDocumentError("FibRgLw is too small to contain ccpText.")
    ccp_text = _i32(fib_lw, FIB97_CCP_TEXT_OFFSET)
    if ccp_text < 0:
        raise InvalidWordDocumentError(f"Invalid negative ccpText: {ccp_text}")

    cb_rg_fc_lcb = _u16(word_document, pos)
    pos += 2
    fib_fc_lcb_size = cb_rg_fc_lcb * 8
    fib_fc_lcb_blob = _slice_exact(word_document, pos, fib_fc_lcb_size, "FibRgFcLcb")

    fc_clx_offset = FIB97_FC_CLX_PAIR_INDEX * 8
    if len(fib_fc_lcb_blob) < fc_clx_offset + 8:
        raise InvalidWordDocumentError("FibRgFcLcb is too small to contain fcClx/lcbClx.")

    fc_clx = _u32(fib_fc_lcb_blob, fc_clx_offset)
    lcb_clx = _u32(fib_fc_lcb_blob, fc_clx_offset + 4)
    if lcb_clx <= 0:
        raise InvalidWordDocumentError("CLX is missing or empty.")

    return FibInfo(
        n_fib=n_fib,
        flags=flags,
        table_stream_name=table_stream_name,
        ccp_text=ccp_text,
        fc_clx=fc_clx,
        lcb_clx=lcb_clx,
    )


def _read_text_pieces(table_stream: bytes, fib: FibInfo) -> list[TextPiece]:
    clx = _slice_exact(table_stream, fib.fc_clx, fib.lcb_clx, "CLX")
    pos = 0

    while pos < len(clx) and clx[pos] == CLX_PRL_MARKER:
        cb_grpprl = _u16(clx, pos + 1)
        pos += 3 + cb_grpprl

    if pos >= len(clx):
        raise InvalidWordDocumentError("CLX ended before a Pcdt marker was found.")
    if clx[pos] != CLX_PCDT_MARKER:
        raise InvalidWordDocumentError(f"Unexpected CLX marker: 0x{clx[pos]:02X}")

    plc_size = _u32(clx, pos + 1)
    plc_data = _slice_exact(clx, pos + 5, plc_size, "PlcPcd")

    if plc_size < 4 or (plc_size - 4) % 12 != 0:
        raise InvalidWordDocumentError("PlcPcd has an invalid size.")

    piece_count = (plc_size - 4) // 12
    cp_count = piece_count + 1
    cps = struct.unpack_from(f"<{cp_count}I", plc_data, 0)
    pcd_base = cp_count * 4

    pieces: list[TextPiece] = []
    for index in range(piece_count):
        cp_start = cps[index]
        cp_end = cps[index + 1]
        if cp_end < cp_start:
            raise InvalidWordDocumentError("Piece table CPs are not monotonic.")

        pcd_offset = pcd_base + (index * 8)
        fc_raw = _u32(plc_data, pcd_offset + 2)
        is_compressed = bool(fc_raw & 0x40000000)
        fc_value = fc_raw & 0x3FFFFFFF
        file_offset = fc_value // 2 if is_compressed else fc_value

        pieces.append(
            TextPiece(
                cp_start=cp_start,
                cp_end=cp_end,
                file_offset=file_offset,
                is_compressed=is_compressed,
            )
        )

    return pieces


def _decode_main_document_text(
    word_document: bytes,
    pieces: Sequence[TextPiece],
    ccp_text: int,
) -> str:
    if ccp_text == 0:
        return ""

    chunks: list[str] = []
    for piece in pieces:
        if piece.cp_start >= ccp_text:
            break
        if piece.cp_end <= 0:
            continue

        read_cp_start = max(piece.cp_start, 0)
        read_cp_end = min(piece.cp_end, ccp_text)
        if read_cp_start >= read_cp_end:
            continue

        char_skip = read_cp_start - piece.cp_start
        char_count = read_cp_end - read_cp_start

        if piece.is_compressed:
            byte_offset = piece.file_offset + char_skip
            raw = _slice_exact(word_document, byte_offset, char_count, "compressed text piece")
            chunks.append(_decode_compressed_piece(raw))
            continue

        byte_offset = piece.file_offset + (char_skip * 2)
        raw = _slice_exact(word_document, byte_offset, char_count * 2, "unicode text piece")
        try:
            chunks.append(raw.decode("utf-16le"))
        except UnicodeDecodeError as exc:
            raise InvalidWordDocumentError("Unicode text piece is not valid UTF-16LE.") from exc

    return "".join(chunks)


def _decode_compressed_piece(raw: bytes) -> str:
    return "".join(chr(COMPRESSED_ANSI_EXCEPTIONS.get(value, value)) for value in raw)


def _keep_field_results_only(text: str) -> str:
    output: list[str] = []
    field_depth = 0
    keep_current_result = False

    for char in text:
        if char == FIELD_BEGIN:
            field_depth += 1
            keep_current_result = False
            continue
        if char == FIELD_SEPARATOR:
            if field_depth > 0:
                keep_current_result = True
                continue
        if char == FIELD_END:
            if field_depth > 0:
                field_depth -= 1
                keep_current_result = field_depth > 0 and keep_current_result
                continue

        if field_depth == 0 or keep_current_result:
            output.append(char)

    return "".join(output)


def _slice_exact(data: bytes, start: int, size: int, label: str) -> bytes:
    if start < 0 or size < 0:
        raise InvalidWordDocumentError(f"{label} references a negative range.")

    end = start + size
    if end > len(data):
        raise InvalidWordDocumentError(
            f"{label} range {start}:{end} exceeds stream size {len(data)}."
        )

    return data[start:end]


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _i32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract plain text from a legacy Word .doc file.")
    parser.add_argument("file_path", type=Path, help="Path to a legacy Word .doc file.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove Word field instructions and translate common control markers.",
    )
    args = parser.parse_args(argv)

    text = extract_doc_text(args.file_path)
    if args.clean:
        text = cleanup_extracted_text(text)

    sys.stdout.buffer.write(text.encode("utf-8"))
    return 0


__all__ = [
    "OleParserError",
    "NotOleWordDocumentError",
    "EncryptedWordDocumentError",
    "InvalidWordDocumentError",
    "extract_doc_text",
    "cleanup_extracted_text",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
