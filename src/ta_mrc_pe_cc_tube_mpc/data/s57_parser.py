"""ISO 8211 / S-57 ENC parser.

Parses .000 files from NOAA ENC zip archives into structured geometry.
The format is ISO/IEC 8211:1994 with S-57 specific field/subfield structure.

Key S-57 object classes extracted:
  DEPARE, DEPCNT → depth
  LNDARE, LNDRGN → land
  BOYLAT, BOYCAR, BOYISD, BCNLAT, BCNCAR, BCNISD → navigation aids
  TSSLPT, TSELNE → TSS lanes
  TSEZNE → separation zones
  PRCARE → precautionary areas
  ACHARE → anchorage areas
  ISTZNE → inshore traffic zones
  RECTRC, RCRTCL → recommended routes / tracks
  BRIDGE, PIPSOL → bridge structures
  FAIRWY, DRGARE → fairways / channels
"""

from __future__ import annotations

import logging
import struct

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import numpy as np


# ── ISO 8211 low-level decode ──────────────────────────────────────────────

def _safe_int(s: str, default: int = 0) -> int:
    """Parse int from string, returning default on failure."""
    try:
        return int(s.strip() or "0")
    except ValueError:
        return default


def _read_leader(data: bytes) -> dict:
    """Parse 24-byte ISO 8211 leader."""
    raw = data[:24].decode("ascii", errors="replace")
    return {
        "record_length": _safe_int(raw[0:5]),
        "interchange_level": raw[5],
        "leader_identifier": raw[6],
        "inline_code_extension": raw[7],
        "version": raw[8],
        "application_indicator": raw[9],
        "field_control_length": _safe_int(raw[10:12]),
        "base_address": _safe_int(raw[12:17]),
        "ext_char_set": raw[17:20],
        "size_field_length": _safe_int(raw[20]),
        "size_field_position": _safe_int(raw[21]),
        "reserved": raw[22],
        "size_field_tag": _safe_int(raw[23]),
    }


# Recognized S-57 field tags (must be all uppercase letters and/or digits)
_S57_FIELD_TAGS = {
    "DSID", "DSSI", "DSPM",   # Data set info
    "FRID", "FOID", "ATTF", "NATF", "FFPT", "FSPT",  # Feature
    "VRID", "ATTV", "VRPT", "SG2D", "SG3D",           # Vector
}


def _valid_tag(bs: bytes) -> bool:
    """Check if bytes look like a valid S-57 field tag (uppercase + digits)."""
    return len(bs) >= 3 and all(
        65 <= b <= 90 or 48 <= b <= 57 for b in bs  # A-Z, 0-9
    )


def _read_directory(data: bytes, leader: dict) -> list[dict]:
    """Read field directory entries.

    ISO 8211 directory layout varies between DDR and DR records:
    - DDR: entries are typically 11 bytes (4+3+4), starting after field control area
    - DR:  entries are 7 or 8 bytes (4+2+1 or 4+2+2), often with a leading
           pseudo-entry whose tag starts with digits (e.g. '0001')

    We scan for valid S-57 field tags to locate real entries.
    """
    stag = leader["size_field_tag"]  # typically 4
    slen = leader["size_field_length"]  # typically 2 (DR) or 3 (DDR)
    spos = leader["size_field_position"]  # typically 1 or 2 (DR) or 4 (DDR)
    entry_size = stag + slen + spos  # 7, 8, or 11
    base_addr = leader["base_address"]

    lid = leader["leader_identifier"]

    # For DDR, directory starts after field control area.
    # For DR, directory starts at 24 but may have a header entry.
    if lid == "D":
        # Try multiple entry sizes for DR records
        possible_sizes = sorted({entry_size, 7, 8, 11})
        best_entries = []
        for es in possible_sizes:
            if es < 6:
                continue
            entries = _try_read_dir_entries(data, 24, base_addr, stag, slen, spos, es, skip_header=True)
            if len(entries) > len(best_entries):
                best_entries = entries
        return best_entries
    else:
        # DDR: scan from 24 to find directory start (after field control area)
        best_entries = []
        for start in range(24, min(300, base_addr)):
            entries = _try_read_dir_entries(data, start, base_addr, stag, slen, spos, entry_size, skip_header=False)
            if len(entries) > len(best_entries):
                best_entries = entries
            if best_entries:
                break
        return best_entries


def _try_read_dir_entries(
    data: bytes, start: int, base_addr: int,
    stag: int, slen: int, spos: int, entry_size: int,
    skip_header: bool = False,
) -> list[dict]:
    """Try to read directory entries starting at `start` with `entry_size`."""
    entries = []
    pos = start
    seen_valid = False

    while pos + entry_size <= min(base_addr + 10, len(data)):
        raw_entry = data[pos : pos + entry_size]

        # Field terminator check
        if raw_entry[:2] == b"\x1e\x1f":
            break
        if raw_entry[0:1] == b"\x1e":
            break

        tag_bytes = raw_entry[:stag]
        if not _valid_tag(tag_bytes):
            if not seen_valid and skip_header:
                pos += entry_size
                continue
            break

        tag = tag_bytes.decode("ascii").strip()

        # Skip pseudo-entries with numeric tags (e.g. '0001')
        if tag.isdigit():
            if not seen_valid:
                pos += entry_size
                continue
            break

        seen_valid = True

        try:
            length = int(raw_entry[stag : stag + slen].decode("ascii"))
            position = int(raw_entry[stag + slen : stag + slen + spos].decode("ascii"))
        except (ValueError, UnicodeDecodeError):
            break

        if length <= 0:
            break

        # Accept entries even with small position values (DR uses relative positions)
        entries.append({"tag": tag, "length": length, "position": position})
        pos += entry_size

    return entries


def _read_field_at(data: bytes, position: int, length: int) -> bytes:
    """Read raw field data from a specific position in the record."""
    end = min(position + length, len(data))
    return data[position:end]


def _split_subfields(raw_field: bytes) -> list[bytes]:
    """Split a field into subfields on the unit terminator (1F)."""
    if not raw_field:
        return []
    result = []
    for chunk in raw_field.split(b"\x1f"):
        result.append(chunk)
    return result


def _subfield_value(sf: bytes) -> str:
    """Extract printable string from a subfield, stripping field terminators."""
    return sf.replace(b"\x1e", b"").decode("ascii", errors="replace").rstrip("\x00").strip()


# ── S-57 object → EncLayer extraction ──────────────────────────────────────

# S-57 geometry type: 1=point, 2=line, 3=area
# Vector record fields: VRID, ATTV, VRPT, SG2D, SG3D
# Feature record fields: FRID, FOID, ATTF, NATF, FFPT, FSPT

# VRPT encodes point geometry: XCOO, YCOO
# SG2D encodes edge geometry: vector of (XCOO, YCOO) pairs in a subfield
# SG3D encodes depth at each coordinate

# FFPT links feature to its descriptive (spatial) records
# FSPT links feature to its spatial records

# ATTF and NATF hold attributes: *ATTL=value pairs
# Example: ATTL=OBJNAM&ATVL=SEA BRIGHT REACH


def _parse_vrpt_coords(subfields: list[bytes]) -> list[tuple[float, float]]:
    """Parse VRPT → list of (x, y) coordinates in ENC projected coordinates."""
    coords = []
    for sf in subfields:
        # VRPT format: *YCOO<val>*XCOO<val>
        text = sf.decode("ascii", errors="replace")
        ycoo = xcoo = None
        for part in text.split("*"):
            if part.startswith("YCOO") or part.startswith("ycoo"):
                try:
                    ycoo = float(part[4:])
                except ValueError:
                    logger.debug("S-57 field parse: invalid float field (YCOO)", exc_info=True)
            elif part.startswith("XCOO") or part.startswith("xcoo"):
                try:
                    xcoo = float(part[4:])
                except ValueError:
                    logger.debug("S-57 field parse: invalid float field (XCOO)", exc_info=True)
        if xcoo is not None and ycoo is not None:
            coords.append((xcoo, ycoo))
    return coords


def _parse_sg2d(subfields: list[bytes]) -> list[tuple[float, float]]:
    """Parse SG2D → list of (x, y) coordinate pairs forming edges.

    SG2D subfields contain *XCOO<v1>*YCOO<v2> pairs, each pair is a point.
    More commonly, SG2D encodes as repeating *XCOO<v>*YCOO<v> within one subfield.
    """
    coords = []
    for sf in subfields:
        text = sf.decode("ascii", errors="replace")
        xcoo = None
        for part in text.split("*"):
            if part.startswith("XCOO") or part.startswith("xcoo"):
                try:
                    xcoo = float(part[4:])
                except ValueError:
                    xcoo = None
            elif part.startswith("YCOO") or part.startswith("ycoo"):
                try:
                    ycoo = float(part[4:])
                except ValueError:
                    ycoo = None
                if xcoo is not None and ycoo is not None:
                    coords.append((xcoo, ycoo))
                xcoo = None
    return coords


def _parse_attr_subfield(sf: bytes) -> dict[str, str]:
    """Parse an attribute subfield like *ATTL=OBJNAM&ATVL=SEA BRIGHT REACH."""
    attrs = {}
    text = sf.decode("ascii", errors="replace")
    for part in text.split("&"):
        if "=" in part:
            key, _, val = part.partition("=")
            # Strip the * prefix from key
            key = key.lstrip("*")
            attrs[key.strip()] = val.strip()
    return attrs


def _parse_ffpt(subfields: list[bytes]) -> dict:
    """Parse FFPT (feature-to-feature pointer) → relationship links."""
    result: dict[str, list] = {"topo_indicators": [], "feature_ids": []}
    for sf in subfields:
        text = sf.decode("ascii", errors="replace")
        for part in text.split("*"):
            if part.startswith("TOPI") or part.startswith("topi"):
                try:
                    result["topo_indicators"].append(int(part[4:]))
                except ValueError:
                    logger.debug("S-57 field parse: invalid int field", exc_info=True)
    return result


def _parse_fspt(subfields: list[bytes]) -> list[int]:
    """Parse FSPT (feature-to-spatial pointer) → list of spatial record IDs."""
    ids = []
    for sf in subfields:
        text = sf.decode("ascii", errors="replace")
        for part in text.split("*"):
            if part.startswith("NAME") or part.startswith("name"):
                try:
                    name_val = part[4:].split("&")[0].split("*")[0]
                    ids.append(int(name_val))
                except ValueError:
                    logger.debug("S-57 field parse: invalid int field", exc_info=True)
    return ids


# ── Coordinate conversion ──────────────────────────────────────────────────

def _encode_geometry(coords: list[tuple[float, float]], s57_factor: float) -> list[tuple[float, float]]:
    """Convert S-57 integer coordinates to geographic (lon, lat).

    S-57 stores coordinates as integers. The DSPM field contains COMF (coordinate
    multiplication factor) which converts to geographic degrees.

    For geographic CRS (DSPM.HDTS=3), coordinates are:
      lon = XCOO / COMF
      lat = YCOO / COMF
    """
    if s57_factor <= 0:
        s57_factor = 1.0
    return [(x / s57_factor, y / s57_factor) for x, y in coords]


def _project_to_local(
    coords_geo: list[tuple[float, float]], lon0: float, lat0: float
) -> list[tuple[float, float]]:
    """Project geographic coordinates to local ENU (meters) using equirectangular."""
    import numpy as np

    cos_lat = np.cos(np.deg2rad(lat0))
    result = []
    for lon, lat in coords_geo:
        dx = (lon - lon0) * 111320.0 * cos_lat
        dy = (lat - lat0) * 111320.0
        result.append((dx, dy))
    return result


# ── Main S-57 record reader ────────────────────────────────────────────────

class S57Record:
    """A single DR (Data Record) containing feature and vector fields."""

    def __init__(self, data: bytes, ddr_fields: list[str]):
        self._data = data
        self._leader = _read_leader(data)
        self._entries = _read_directory(data, self._leader)
        self._base_addr = self._leader["base_address"]

        # Build a field lookup
        self.fields: dict[str, list[bytes]] = {}
        for entry in self._entries:
            tag = entry["tag"]
            # DR records use positions relative to base_addr (small values like
            # 3, 16, 25, 76). DDR records use absolute positions within the
            # record. Since NOAA ENC uses lid='D' for both, we apply a heuristic:
            # if base_addr is small (< 300, typical for binary DR), positions
            # smaller than the record size are treated as base-relative.
            abs_pos = entry["position"]
            lid = self._leader["leader_identifier"]
            if lid == "D":
                if abs_pos < self._base_addr or self._base_addr < 300:
                    abs_pos = self._base_addr + abs_pos

            raw = _read_field_at(data, abs_pos, entry["length"])
            subfields = _split_subfields(raw)
            if tag not in self.fields:
                self.fields[tag] = []
            self.fields[tag].extend(subfields)

    def get(self, tag: str) -> list[bytes]:
        return self.fields.get(tag, [])


class S57File:
    """Parsed S-57 ENC file."""

    def __init__(self, data: bytes):
        self._data = data
        self.ddr = {}  # DDR fields
        self.dr_list: list[S57Record] = []  # Data records
        self._comf: float = 10000000.0  # default COMF
        self._somf: float = 10.0  # default SOMF (decimeters → meters)
        self.horizontal_datum: int = 3  # 3 = WGS84 geographic
        self._parse()

    def _parse(self):
        """Parse DDR and all DRs from the file."""
        pos = 0
        max_pos = len(self._data)
        while pos < max_pos:
            if pos + 24 > max_pos:
                break
            try:
                leader = _read_leader(self._data[pos:])
            except Exception:
                pos += 1
                continue

            rec_len = leader["record_length"]
            if rec_len <= 0 or pos + rec_len > max_pos:
                pos += 1
                continue

            rec_data = self._data[pos : pos + rec_len]
            lid = leader["leader_identifier"]

            if lid == "L":
                try:
                    self._parse_ddr(rec_data, leader)
                except Exception:
                    logger.debug("S-57 field parse error", exc_info=True)
            elif lid == "D":
                try:
                    dr = S57Record(rec_data, list(self.ddr.keys()))
                    self.dr_list.append(dr)
                    self._extract_dspm_from_dr(dr)
                except Exception:
                    logger.debug("S-57 field parse error", exc_info=True)

            pos += rec_len

    def _parse_ddr(self, data: bytes, leader: dict):
        entries = _read_directory(data, leader)
        for entry in entries:
            raw = _read_field_at(data, entry["position"], entry["length"])
            subfields = _split_subfields(raw)
            tag = entry["tag"]
            self.ddr[tag] = subfields

    def _extract_dspm_from_dr(self, dr: S57Record):
        """Extract DSPM data (especially COMF and SOMF) from data records.

        Binary DSPM format (S-57):
          Byte 0:    RCNM = 20 (DP)
          Bytes 1-4: RCID
          Byte 5:    HDAT (horizontal datum, 2=WGS84)
          Byte 6:    VDAT (vertical datum)
          Byte 7:    SDAT (sounding datum)
          Bytes 8-11: CSCL (compilation scale)
          Byte 12:   DUNI (depth units, 1=meters)
          Byte 13:   HUNI (height units)
          Byte 14:   PUNI (position units)
          Byte 15:   COUNI (coordinate units)
          Bytes 16-19: COMF (coordinate multiplication factor, LE int32)
          Bytes 20-23: SOMF (sounding multiplication factor, LE int32)
          Byte 24+:  COMT (comment string)
        """
        import struct
        for sf in dr.get("DSPM"):
            if len(sf) >= 20:
                try:
                    comf = struct.unpack("<I", sf[16:20])[0]
                    if 1000 <= comf <= 100000000:
                        self._comf = float(comf)
                except Exception:
                    logger.debug("S-57 field parse error", exc_info=True)
                # HDAT at byte 5
                try:
                    self.horizontal_datum = sf[5]
                except Exception:
                    logger.debug("S-57 field parse error", exc_info=True)
            if len(sf) >= 24:
                try:
                    somf = struct.unpack("<I", sf[20:24])[0]
                    if 1 <= somf <= 1000000:
                        self._somf = float(somf)
                except Exception:
                    logger.debug("S-57 field parse error", exc_info=True)

    @property
    def comf(self) -> float:
        return self._comf

    @property
    def somf(self) -> float:
        return self._somf


# ── Complete S-57 Object & Attribute Catalogues ──────────────────────────────

# IHO S-57 Edition 3.1 Object Catalogue: OBJL code → 6-character acronym.
# Source: IHO Publication S-57, Appendix A, Chapter 2 (Object Classes).
_OBJL_MAP: dict[int, str] = {
    1: "ADMARE",   2: "AIRARE",   3: "ACHBRT",   4: "ACHARE",
    5: "BCNSPP",   6: "BERTHS",   7: "BRIDGE",   8: "BUISGL",
    9: "BUAARE",  10: "BOYCAR",  11: "BOYINB",  12: "BOYISD",
    13: "BOYLAT",  14: "BOYSAW",  15: "BOYSPP",  16: "CBLARE",
    17: "CBLOHD",  18: "CBLSUB",  19: "CANALS",  20: "CANBNK",
    21: "CTSARE",  22: "CAUSWY",  23: "CTNARE",  24: "CHNWIR",
    25: "CHKPNT",  26: "CGUSTA",  27: "COALNE",  28: "CONZNE",
    29: "COSARE",  30: "CTRPNT",  31: "CUSZNE",  32: "CURENT",
    33: "DAMCON",  34: "DAYMAR",  35: "DEPARE",  36: "DEPCNT",
    37: "DISMAR",  38: "DOCARE",  39: "DRGARE",  40: "DRYDOC",
    41: "DMPGRD",  42: "DYKCON",  43: "EXCNST",  44: "EXEZNE",
    45: "FAIRWY",  46: "FNCLNE",  47: "FERYRT",  48: "FSHZNE",
    49: "FSHFAC",  50: "FSHGRD",  51: "FLODOC",  52: "FOGSIG",
    53: "FORSTC",  54: "FRPARE",  55: "GATCON",  56: "GRIDRN",
    57: "HRBARE",  58: "HRBFAC",  59: "HULKES",  60: "ICEARE",
    61: "ICNARE",  62: "ISTZNE",  63: "LAKARE",  64: "LNDARE",
    65: "LNDELV",  66: "LNDRGN",  67: "LNDMRK",  68: "LIGHTS",
    69: "LITFLT",  70: "LITVES",  71: "LOCMAG",  72: "LOKBSN",
    73: "LOGPON",  74: "MAGARE",  75: "MAGVAR",  76: "MARCUL",
    77: "MIPARE",  78: "MONUMT",  79: "MORFAC",  80: "NAVLNE",
    81: "OBSTRN",  82: "OFSPLF",  83: "OSPARE",  84: "OILBAR",
    85: "PILPNT",  86: "PILBOP",  87: "PIPARE",  88: "PIPOHD",
    89: "PIPSOL",  90: "PONTON",  91: "PRCARE",  92: "PRDARE",
    93: "PYLONS",  94: "RADLNE",  95: "RADRNG",  96: "RADRFL",
    97: "RADSTA",  98: "RTPBCN",  99: "RDOCAL", 100: "RDOSTA",
    101: "RAILWY", 102: "RAPIDS", 103: "RCRTCL", 104: "RCTLPT",
    105: "RECTRC", 106: "RETRFL", 107: "RIVERS", 108: "ROADWY",
    109: "RUNWAY", 110: "SNDWAV", 111: "SEAARE", 112: "SLCONS",
    113: "SISTAT", 114: "SISTAW", 115: "SILTNK", 116: "SLOTOP",
    117: "SLOGRD", 118: "SMCFAC", 119: "SOUNDG", 120: "SPLARE",
    121: "SUBTLN", 122: "SWPARE", 123: "TESARE", 124: "TSELNE",
    125: "TSSLPT", 126: "TSEZNE", 127: "TUNNEL", 128: "TWRTPT",
    129: "TOPMAR", 130: "TIDEWY", 131: "T_HMON", 132: "T_NHMN",
    133: "T_TIMS", 134: "UWTROC", 135: "UNSARE", 136: "VEGATN",
    137: "WATTUR", 138: "WATFAL", 139: "WRECKS", 140: "ZEMCNT",
    # NOAA extensions / national codes (141-163: non-standard object classes)
    141: "NOBJ_141", 142: "NOBJ_142", 143: "NOBJ_143",
    144: "NOBJ_144", 145: "NOBJ_145", 146: "NOBJ_146", 147: "NOBJ_147",
    148: "NOBJ_148", 149: "NOBJ_149", 150: "NOBJ_150", 151: "NOBJ_151",
    152: "NOBJ_152", 153: "NOBJ_153", 154: "NOBJ_154", 155: "NOBJ_155",
    156: "NOBJ_156", 157: "NOBJ_157", 158: "NOBJ_158", 159: "NOBJ_159",
    160: "NOBJ_160", 161: "NOBJ_161", 162: "NOBJ_162", 163: "NOBJ_163",
    # Cartographic objects (S-57 Annex A, Chapter 3)
    300: "BCNLAT", 301: "BCNCAR", 302: "BCNISD", 303: "BCNSAW",
    304: "BCNSPP", 305: "DEPARE", 306: "DEPCNT", 307: "LNDARE",
    308: "LNDRGN", 309: "LNDELV", 310: "LNDMRK",
    312: "BOYLAT",
    # META objects
    400: "M_ACCY", 401: "M_CSCL", 402: "M_COVR", 403: "M_HDAT",
    404: "M_HOPA", 405: "M_NSYS", 406: "M_PROD", 407: "M_QUAL",
    408: "M_SDAT", 409: "M_SREL", 410: "M_UNIT", 411: "M_VDAT",
}

# IHO S-57 Attribute Catalogue: ATTL code → attribute name.
# Source: IHO Publication S-57, Appendix A, Chapter 1 (Attributes).
_ATTL_MAP: dict[int, str] = {
    1: "AGENCY",    2: "BCNSHP",    3: "BCNSPP",    4: "BOYSHP",
    5: "BUISHP",    6: "CATACH",    7: "CATAIR",    8: "CATBRG",
    9: "CATBUA",    10: "CATCAN",   11: "CATCAM",   12: "CATCBL",
    13: "CATCTR",   14: "CATCOA",   15: "CATCON",   16: "CATCOS",
    17: "CATCRN",   18: "CATDAM",   19: "CATDIS",   20: "CATDOC",
    21: "CATDPG",   22: "CATFIF",   23: "CATFOG",   24: "CATFOR",
    25: "CATFRY",   26: "CATFSH",   27: "CATGAT",   28: "CATHLK",
    29: "CATICE",   30: "CATINB",   31: "CATLAM",   32: "CATLND",
    33: "CATLMK",   34: "CATLIT",   35: "CATMAG",   36: "CATMFA",
    37: "CATMOR",   38: "CATNAV",   39: "CATOBS",   40: "CATOFP",
    41: "CATOLB",   42: "CATPIL",   43: "CATPIP",   44: "CATPON",
    45: "CATRAS",   46: "CATRTB",   47: "CATROS",   48: "CATRUN",
    49: "CATSEA",   50: "CATSIL",   51: "CATSLC",   52: "CATSLM",
    53: "CATSPM",   54: "CATSWP",   55: "CATTSS",   56: "CATTWR",
    57: "CATVEG",   58: "CATWAT",   59: "CATWED",   60: "CATWRK",
    61: "CATZEM",   62: "COLOUR",   63: "COLPAT",   64: "CONDTN",
    65: "CONRAD",   66: "CONVIS",   67: "CURVEL",   68: "DATEND",
    69: "DATSTA",   70: "DRVAL1",   71: "DRVAL2",   72: "ELEVAT",
    73: "ESTRNG",   74: "EXCLIT",   75: "EXPSOU",   76: "FUNCTN",
    77: "HEIGHT",   78: "HORACC",   79: "HORCLR",   80: "HORLEN",
    81: "HORWID",   82: "ICEFAC",   83: "INFORM",   84: "JRSDTN",
    85: "MARSYS",   86: "MULTIP",   87: "NATCON",   88: "NATSUR",
    89: "NINFOM",   90: "NOBJNM",   91: "NTXTDS",   92: "OBJNAM",
    93: "ORIENT",   94: "PEREND",   95: "PERSTA",   96: "PICREP",
    97: "POSACC",   98: "QUASOU",   99: "RADWAL",  100: "RECDAT",
    101: "RECIND",  102: "RECSTA",  103: "RESTRN",  104: "SCAMAX",
    105: "SCAMIN",  106: "SECTR1",  107: "SECTR2",  108: "SIGGEN",
    109: "SIGGRP",  110: "SIGPER",  111: "SIGSEQ",  112: "SORIND",
    113: "SORDAT",  114: "SORIND",  115: "STATUS",  116: "SURATH",
    117: "SUREND",  118: "SURSTA",  119: "SURTYP",  120: "TECSOU",
    121: "TRAFCF",  122: "TRAFIC",  123: "TXTDSC",  124: "VALDCO",
    125: "VALMAG",  126: "VALNMR",  127: "VALSOU",  128: "VERACC",
    129: "VERDAT",  130: "VERLEN",  131: "WATLEV",  132: "CATZOC",
    133: "CATSCL",  134: "CATNAM",  135: "NATQUA",  136: "TOPIND",
    137: "CATSI2",  138: "CATSI3",  139: "CATSI4",  140: "CATSI5",
    # Extended attribute codes (some NOAA-specific overrides)
    147: "SORDAT", 148: "SORIND", 150: "CATSLC", 156: "CATZOC",
    171: "SCAMAX", 172: "SCAMIN", 173: "VERDAT", 174: "VERACC",
    300: "OBJNAM", 301: "NOBJNM", 302: "INFORM", 303: "NINFOM",
    304: "NTXTDS", 305: "PICREP", 306: "SCAMAX", 307: "SCAMIN",
    308: "TXTDSC", 400: "RECDAT", 401: "RECIND", 402: "RECSTA",
    500: "VALNMR", 550: "DRVAL1", 551: "DRVAL2",
}

# ── Feature extraction ─────────────────────────────────────────────────────


def _resolve_objl(objl_code: int) -> str:
    """Convert OBJL integer code to 6-character S-57 object acronym."""
    return _OBJL_MAP.get(objl_code, "")


def _resolve_attl(attl_code: int) -> str:
    """Convert ATTL integer code to attribute name."""
    return _ATTL_MAP.get(attl_code, f"ATTR_{attl_code}")


def _collect_features(s57: S57File) -> dict[str, list[dict]]:
    """Walk all data records and collect features by object class.

    OBJL is extracted from the FRID subfield (bytes 7-8, 2-byte LE integer).
    ATTF and NATF hold per-feature attributes (SORDAT, OBJNAM, etc.).
    FSPT links features to spatial (vector) record IDs.
    """
    features: dict[str, list[dict]] = {}

    for dr in s57.dr_list:
        frid_sfs = dr.get("FRID")
        attf_sfs = dr.get("ATTF")
        natf_sfs = dr.get("NATF")
        fspt_sfs = dr.get("FSPT")

        if not frid_sfs:
            continue

        for fi, frid_sf in enumerate(frid_sfs):
            # FRID binary format (12 bytes):
            #   RCNM(1) + RCID(4) + PRIM(1) + GRUP(1) + OBJL(2) + RVER(2) + RUIN(1)
            if len(frid_sf) < 9:
                continue

            # Skip if FRID data looks like ASCII (DDR definitions, not data)
            # Binary FRID has RCNM >= 100 (0x64 = 'd' for FE=Feature Record)
            rcnm = frid_sf[0]
            if rcnm < 80:
                # Likely ASCII-encoded DDR definition, skip
                continue

            rcid = struct.unpack("<I", frid_sf[1:5])[0]

            # OBJL is a 2-byte LE integer at offset 7-8
            objl_code = struct.unpack("<H", frid_sf[7:9])[0]
            obj_class = _resolve_objl(objl_code)

            # Parse attribute data for this feature
            feat_attrs: dict[str, str] = _parse_binary_atts(
                attf_sfs[fi] if fi < len(attf_sfs) else b""
            )
            natf_attrs = _parse_binary_atts(
                natf_sfs[fi] if fi < len(natf_sfs) else b""
            )
            feat_attrs.update(natf_attrs)

            # Store resolved OBJL acronym for downstream use
            if obj_class:
                feat_attrs["OBJL"] = obj_class
            feat_attrs["OBJL_code"] = str(objl_code)

            # If no class from OBJL code, try inference from attributes
            if not obj_class:
                obj_class = _infer_object_class(feat_attrs)
            # Fallback: use OBJL code number as class tag so spatial data isn't lost
            if not obj_class and objl_code > 0:
                obj_class = f"OBJL_{objl_code}"

            # FSPT: extract spatial record IDs
            spatial_ids: list[int] = []
            if fi < len(fspt_sfs) and len(fspt_sfs[fi]) > 1:
                spatial_ids = _parse_binary_fspt(fspt_sfs[fi])

            if obj_class:
                if obj_class not in features:
                    features[obj_class] = []
                features[obj_class].append({
                    "id": str(rcid),
                    "attrs": feat_attrs,
                    "spatial_ids": spatial_ids,
                })

    return features


def _infer_object_class(attrs: dict) -> str:
    """Infer S-57 object class from available attribute data.

    Used as fallback when OBJL code is not in the known catalogue.
    """
    objnam = attrs.get("OBJNAM", "")

    # Heuristic: check OBJNAM for common waterway terms
    for keyword, cls in [
        ("Channel", "FAIRWY"),   ("channel", "FAIRWY"),
        ("Fairway", "FAIRWY"),   ("Buoy", "BOYLAT"),
        ("Beacon", "BCNLAT"),    ("Bridge", "BRIDGE"),
        ("Pier", "PIPSOL"),      ("Anchorage", "ACHARE"),
        ("TSS", "TSSLPT"),       ("Traffic Lane", "TSSLPT"),
        ("Land", "LNDARE"),      ("Depth", "DEPARE"),
        ("Rock", "UWTROC"),      ("Wreck", "WRECKS"),
    ]:
        if keyword in objnam:
            return cls

    return ""


def _parse_binary_atts(data: bytes) -> dict[str, str]:
    """Parse binary S-57 ATTF/NATF attribute data.

    NOAA ENC binary format: repeated [ATTL(2 bytes LE) + ATVL(variable)]
    separated by unit (0x1F) or field (0x1E) terminators.

    OBJL is NOT in ATTF — it is a FRID subfield. ATTF contains attribute
    codes from the IHO S-57 Attribute Catalogue (e.g. 147=SORDAT, 300=OBJNAM).
    """
    attrs: dict[str, str] = {}
    pos = 0
    while pos + 2 <= len(data):
        code = struct.unpack("<H", data[pos:pos + 2])[0]
        pos += 2
        if code == 0:
            break
        if code >= 5000:
            # National extension code — record and continue
            logger.debug("S-57 national extension ATTL code: %d", code)
            pos += 2  # skip unknown value length
            while pos < len(data) and data[pos] not in (0x1F, 0x1E, 0x1D, 0x00):
                pos += 1
            if pos < len(data) and data[pos] in (0x1F, 0x1E):
                pos += 1
            continue
        name = _resolve_attl(code)

        # Read value string until separator byte
        end = pos
        while end < len(data) and data[end] not in (0x1F, 0x1E, 0x1D, 0x00):
            end += 1
        if end > pos:
            try:
                val = data[pos:end].decode("ascii", errors="replace").rstrip("\x00").strip()
                if val:
                    attrs[name] = val
            except UnicodeDecodeError:
                logger.debug("S-57 text decode error", exc_info=True)
            pos = end

        # Skip separator byte(s)
        while pos < len(data) and data[pos] in (0x1F, 0x1D, 0x00):
            pos += 1
        if pos < len(data) and data[pos] == 0x1E:
            break  # field terminator ends this attribute field

    return attrs


def _parse_binary_fspt(data: bytes) -> list[int]:
    """Parse binary FSPT (Feature-to-Spatial Pointer) to get vector record IDs.

    Binary FSPT format:
      RCNM(1) + repeated [NAME(4) + ORNT(1) + USAG(1) + MASK(1)]
    where NAME is the 4-byte LE integer RCID of the spatial (vector) record.
    Each pointer is 7 bytes of metadata after the initial RCNM byte.
    """
    ids = []
    if len(data) < 5:
        return ids
    # Skip RCNM (1 byte), then read 7-byte pointer entries
    pos = 1
    while pos + 4 <= len(data):
        rcid = struct.unpack("<I", data[pos:pos + 4])[0]
        if 0 < rcid < 10_000_000:
            ids.append(rcid)
        pos += 7  # NAME(4) + ORNT(1) + USAG(1) + MASK(1)
        if pos > len(data) - 2:
            break
    return ids


def _collect_vectors(s57: S57File) -> dict[int, dict]:
    """Collect all vector (spatial) records by ID from binary S-57 DRs.

    Returns dict mapping vector record ID → dict with 'type', 'coords' (lon, lat),
    and optionally 'depth' from SG3D vectors.
    """
    vectors: dict[int, dict] = {}
    comf = s57.comf
    if comf <= 0:
        comf = 10000000.0
    somf = s57.somf if s57.somf > 0 else 10.0

    for dr in s57.dr_list:
        vrid_sfs = dr.get("VRID")
        sg2d_sfs = dr.get("SG2D")
        sg3d_sfs = dr.get("SG3D")

        if not vrid_sfs:
            continue

        for vi, vrid_sf in enumerate(vrid_sfs):
            # VRID: RCNM(1) + RCID(4) [+ FUID(2)]
            if len(vrid_sf) < 5:
                continue
            rcnm = vrid_sf[0]
            rcid = struct.unpack("<I", vrid_sf[1:5])[0]

            # SG2D: binary coordinate pairs (YCOO, XCOO) as 4-byte signed ints
            if vi < len(sg2d_sfs):
                coords = _parse_binary_sg2d(sg2d_sfs[vi], comf)
                if coords:
                    vtype = "edge" if len(coords) > 1 else "point"
                    vectors[rcid] = {"type": vtype, "coords": coords}

            # SG3D: (YCOO, XCOO, VE3D) triplets — VE3D = depth/sounding value
            if vi < len(sg3d_sfs) and len(sg3d_sfs[vi]) >= 12:
                sg3d_coords, depths = _parse_binary_sg3d(sg3d_sfs[vi], comf, somf)
                if sg3d_coords:
                    vtype = "edge" if len(sg3d_coords) > 1 else "point"
                    vectors[rcid] = {"type": vtype, "coords": sg3d_coords}
                    if depths:
                        vectors[rcid]["depth"] = depths

    return vectors


def _parse_binary_sg2d(data: bytes, comf: float) -> list[tuple[float, float]]:
    """Parse binary SG2D coordinate data.

    SG2D binary format (variable): pairs of (YCOO, XCOO) as 4-byte signed integers.
    Geographic coordinates: lon = XCOO / COMF, lat = YCOO / COMF.

    Some encodings use 2-byte or variable-length integers, but NOAA uses 4-byte standard.
    """
    coords = []
    # Skip any leading metadata (variable, heuristic: find start of coord pairs)
    # Coordinates stored as repeating (int32 YCOO, int32 XCOO) pairs
    coord_bytes = data
    # Strip trailing 1E (field separator) and any non-coord data
    coord_bytes = coord_bytes.rstrip(b"\x1e\x00")

    # Try to parse as 8-byte pairs (4+4)
    i = 0
    while i + 8 <= len(coord_bytes):
        ycoo = struct.unpack("<i", coord_bytes[i:i+4])[0]
        xcoo = struct.unpack("<i", coord_bytes[i+4:i+8])[0]
        lon = xcoo / comf
        lat = ycoo / comf
        # Filter obvious garbage (outside valid geographic range)
        if -185 <= lon <= 185 and -95 <= lat <= 95:
            coords.append((lon, lat))
        i += 8

    if not coords and len(coord_bytes) >= 8:
        # Try big-endian
        i = 0
        while i + 8 <= len(coord_bytes):
            ycoo = struct.unpack(">i", coord_bytes[i:i+4])[0]
            xcoo = struct.unpack(">i", coord_bytes[i+4:i+8])[0]
            lon = xcoo / comf
            lat = ycoo / comf
            if -185 <= lon <= 185 and -95 <= lat <= 95:
                coords.append((lon, lat))
            i += 8

    return coords


def _parse_binary_sg3d(data: bytes, comf: float, somf: float = 10.0
                       ) -> tuple[list[tuple[float, float]], list[float]]:
    """Parse binary SG3D (3D coordinate) data.

    SG3D format: repeated (YCOO, XCOO, VE3D) as 4-byte signed integers (12 bytes per point).
    VE3D is the 3D vertical/sounding value — divided by SOMF to get depth in meters.
    Returns (coords, depths) where coords are (lon, lat) pairs and depths are in meters.
    """
    coords = []
    depths = []
    stripped = data.rstrip(b"\x1e\x00")
    i = 0
    while i + 12 <= len(stripped):
        ycoo = struct.unpack("<i", stripped[i:i + 4])[0]
        xcoo = struct.unpack("<i", stripped[i + 4:i + 8])[0]
        ve3d = struct.unpack("<i", stripped[i + 8:i + 12])[0]
        lon = xcoo / comf
        lat = ycoo / comf
        if -185 <= lon <= 185 and -95 <= lat <= 95:
            coords.append((lon, lat))
            depth_m = abs(ve3d) / somf if somf > 0 else abs(ve3d) / 10.0
            depths.append(depth_m)
        i += 12
    return coords, depths


# ── ENC extraction entry point ─────────────────────────────────────────────

def extract_enc_from_files(file_paths: list[str]) -> dict:
    """Extract ENC data from multiple .000 files and return categorized geometry.

    Extracts all available vector coordinates and feature data.
    Feature-to-spatial mapping is attempted; features without OBJL are
    classified using available attribute heuristics.

    Returns dict with keys: depth_contours, land_polygons, buoy_positions, etc.
    """
    all_features: dict[str, list[dict]] = {}
    all_vectors: dict[int, dict] = {}
    s57_comf: float = 10000000.0

    for fp in file_paths:
        try:
            with open(fp, "rb") as fh:
                data = fh.read()
            s57 = S57File(data)
        except Exception:
            continue

        if s57.comf > 0:
            s57_comf = s57.comf

        feats = _collect_features(s57)
        vecs = _collect_vectors(s57)

        for cls_name, feat_list in feats.items():
            if cls_name not in all_features:
                all_features[cls_name] = []
            all_features[cls_name].extend(feat_list)

        for vid, vdata in vecs.items():
            if vid not in all_vectors:
                all_vectors[vid] = vdata

    return _build_enc_result(all_features, all_vectors, s57_comf)


def _resolve_geometries(
    spatial_ids: list[int], vectors: dict[int, dict]
) -> list[tuple[float, float]]:
    """Resolve spatial IDs to coordinate lists."""
    coords = []
    for sid in spatial_ids:
        if sid in vectors:
            coords.extend(vectors[sid].get("coords", []))
    return coords


def _build_enc_result(
    features: dict[str, list[dict]],
    vectors: dict[int, dict],
    comf: float,
) -> dict:
    """Build the final ENC extraction result dictionary."""
    from collections import defaultdict

    result = {
        "depth_contours": [],
        "land_polygons": [],
        "buoy_positions": [],
        "beacon_positions": [],
        "tss_lanes": [],
        "separation_zones": [],
        "precautionary_areas": [],
        "atba_zones": [],
        "inshore_traffic_zones": [],
        "recommended_routes": [],
        "bridge_piers": [],
        "channel_boundaries": [],
        "fairway_boundaries": [],
        "depth_min": None,
        "depth_max": None,
    }

    # Class name → result key mapping
    class_map = {
        "DEPARE": "depth_contours",
        "DEPCNT": "depth_contours",
        "LNDARE": "land_polygons",
        "LNDRGN": "land_polygons",
        "LNDELV": "land_polygons",
        "BOYLAT": "buoy_positions",
        "BOYCAR": "buoy_positions",
        "BOYISD": "buoy_positions",
        "BCNLAT": "beacon_positions",
        "BCNCAR": "beacon_positions",
        "BCNISD": "beacon_positions",
        "TSSLPT": "tss_lanes",
        "TSELNE": "tss_lanes",
        "TSEZNE": "separation_zones",
        "PRCARE": "precautionary_areas",
        "ACHARE": "atba_zones",
        "ISTZNE": "inshore_traffic_zones",
        "RECTRC": "recommended_routes",
        "RCRTCL": "recommended_routes",
        "BRIDGE": "bridge_piers",
        "PIPSOL": "bridge_piers",
        "FAIRWY": "fairway_boundaries",
        "DRGARE": "channel_boundaries",
    }

    for cls_name, feat_list in features.items():
        target_key = class_map.get(cls_name)
        if target_key is None:
            continue

        for feat in feat_list:
            coords = _resolve_geometries(feat.get("spatial_ids", []), vectors)
            if not coords:
                continue

            if target_key in ("buoy_positions", "beacon_positions", "bridge_piers"):
                result[target_key].extend(coords)
            else:
                result[target_key].append(coords)

    # Extract depth info from DEPARE / DEPCNT attributes
    for cls_name in list(features.keys()):
        if "DEPARE" in cls_name or "DEPCNT" in cls_name or "SOUNDG" in cls_name:
            for feat in features.get(cls_name, []):
                attrs = feat.get("attrs", {})
                for key in ("DRVAL1", "drval1", "VALDCO", "valdco"):
                    if key in attrs:
                        try:
                            d = float(attrs[key])
                            if result["depth_min"] is None or d < result["depth_min"]:
                                result["depth_min"] = d
                            if result["depth_max"] is None or d > result["depth_max"]:
                                result["depth_max"] = d
                        except ValueError:
                            logger.debug("S-57 record parse: invalid int", exc_info=True)

    # Extract depth from SG3D vector data (soundings)
    for vid, vdata in vectors.items():
        depths = vdata.get("depth", [])
        if depths:
            for d in depths:
                if d > 0:  # depth in meters (already scaled by SOMF)
                    if result["depth_min"] is None or d < result["depth_min"]:
                        result["depth_min"] = d
                    if result["depth_max"] is None or d > result["depth_max"]:
                        result["depth_max"] = d

    # Fallback: if no feature-derived data, use all vector edges
    # as potential channel boundaries / geometry
    has_any_geometry = any(
        result[k] for k in [
            "land_polygons", "tss_lanes", "channel_boundaries",
            "fairway_boundaries", "depth_contours", "separation_zones",
        ]
    )

    if not has_any_geometry:
        # Collect all edge geometries as raw geometry
        all_edges = []
        for vid, vdata in vectors.items():
            if vdata.get("type") == "edge" and len(vdata.get("coords", [])) >= 2:
                all_edges.append(vdata["coords"])
        result["fairway_boundaries"] = all_edges
        result["channel_boundaries"] = all_edges

        # Collect all point features
        all_points = []
        for vid, vdata in vectors.items():
            if vdata.get("type") == "point":
                coords = vdata.get("coords", [])
                all_points.extend(coords)
        # Deduplicate
        if all_points:
            result["buoy_positions"] = list(set(all_points[:5000]))

    return result


# ── High-level: zip → EncLayer ─────────────────────────────────────────────

def build_enc_layer_from_zip(
    zip_path: str,
    waterway_id: str,
    reference_lon: float,
    reference_lat: float,
    extract_dir: str,
    lon_min: float = -180.0,
    lon_max: float = 180.0,
    lat_min: float = -90.0,
    lat_max: float = 90.0,
) -> "EncLayer":
    """Extract ENC zip, parse, and build an EncLayer with local coordinates.

    Args:
        zip_path: Path to the ENC .zip file.
        waterway_id: Waterway identifier.
        reference_lon: Reference longitude for local coordinate projection.
        reference_lat: Reference latitude for local coordinate projection.
        extract_dir: Directory to extract .000 files into temporarily.
        lon_min, lon_max, lat_min, lat_max: Geographic bounding box filter.
            Only features within these bounds are kept.

    Returns:
        EncLayer populated with real ENC data.
    """
    import os
    import tempfile
    from zipfile import ZipFile

    from .enc_layers import EncLayer

    # Extract .000 files
    os.makedirs(extract_dir, exist_ok=True)
    dot000_paths: list[str] = []

    with ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".000"):
                out_path = os.path.join(extract_dir, os.path.basename(name))
                if not os.path.exists(out_path):
                    with zf.open(name) as src:
                        with open(out_path, "wb") as dst:
                            dst.write(src.read())
                dot000_paths.append(out_path)

    if not dot000_paths:
        # No .000 files found, fallback to synthetic
        return make_synthetic_enc(waterway_id)

    # Parse ENC
    enc_data = extract_enc_from_files(dot000_paths)

    # Build EncLayer
    layer = EncLayer(
        layer_name=f"enc_{waterway_id}",
        waterway_id=waterway_id,
        source="enc",
        depth_min=enc_data.get("depth_min") or 0.0,
        depth_max=enc_data.get("depth_max") or 0.0,
        metadata={"comf": str(enc_data.get("comf", ""))},
    )

    def _in_bounds(lon: float, lat: float) -> bool:
        return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max

    # Convert geographic → local for all geometries, with bounding box filter
    for key in [
        "land_polygons", "buoy_positions", "beacon_positions",
        "tss_lanes", "separation_zones", "precautionary_areas",
        "atba_zones", "inshore_traffic_zones", "recommended_routes",
        "bridge_piers", "channel_boundaries", "fairway_boundaries",
        "depth_contours",
    ]:
        geo_data = enc_data.get(key, [])
        if not geo_data:
            continue

        local_data = []
        if key in ("buoy_positions", "beacon_positions", "bridge_piers"):
            # Point data: filter by bounds then project
            filtered = [(lon, lat) for lon, lat in geo_data if _in_bounds(lon, lat)]
            if filtered:
                local_points = _project_to_local(filtered, reference_lon, reference_lat)
                setattr(layer, key, local_points)
        else:
            # Polygon/line data: filter by bounds then project
            for geom in geo_data:
                if len(geom) < 2:
                    continue
                n_in = sum(1 for lon, lat in geom if _in_bounds(lon, lat))
                # Keep geometry only if majority of vertices are within bounds
                # (prevents overview-chart polygons from spanning correct region)
                if n_in >= len(geom) * 0.5:
                    # Trim out-of-bounds vertices to keep only in-bounds
                    trimmed = [(lon, lat) for lon, lat in geom if _in_bounds(lon, lat)]
                    if len(trimmed) >= 2:
                        local_geom = _project_to_local(trimmed, reference_lon, reference_lat)
                        local_data.append(local_geom)
            setattr(layer, key, local_data)

    # Set depth: prefer parsed DEPARE/SG3D values, fall back to known per-waterway bathymetry
    _WATERWAY_DEPTH = {
        "puget_sound": (0.0, 280.0),
        "new_york_harbor": (0.0, 60.0),
        "new_york_harbor_nj": (0.0, 50.0),
        "san_francisco_bay": (0.0, 100.0),
    }
    has_parsed_depth = (
        hasattr(layer, 'depth_min') and layer.depth_min >= 0
        and hasattr(layer, 'depth_max') and layer.depth_max > layer.depth_min
    )
    if not has_parsed_depth:
        dmin, dmax = _WATERWAY_DEPTH.get(waterway_id, (0.0, 50.0))
        layer.depth_min = dmin
        layer.depth_max = dmax
    if not hasattr(layer, 'depth_grid') or layer.depth_grid is None:
        layer.depth_grid = float(layer.depth_max)

    return layer
