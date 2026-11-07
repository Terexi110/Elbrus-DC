"""Minimal ELF64 little-endian reader: sections, symbols, RELA relocations.

Only what the decompiler needs: names for addresses, bytes for string
literals and data, relocations inside data sections (pointer tables).
"""
import struct

SHT_SYMTAB = 2
SHT_RELA = 4
SHT_NOBITS = 8
SHT_DYNSYM = 11

STT_NOTYPE, STT_OBJECT, STT_FUNC, STT_SECTION = 0, 1, 2, 3


class Section:
    def __init__(self, idx, name, typ, flags, addr, off, size, link, info, entsize):
        self.idx, self.name, self.type, self.flags = idx, name, typ, flags
        self.addr, self.offset, self.size = addr, off, size
        self.link, self.info, self.entsize = link, info, entsize

    def __repr__(self):
        return f"<Section {self.name} @{self.addr:#x} size={self.size:#x}>"


class Symbol:
    def __init__(self, name, value, size, typ, bind, shndx):
        self.name, self.value, self.size = name, value, size
        self.type, self.bind, self.shndx = typ, bind, shndx

    def __repr__(self):
        return f"<Sym {self.name} {self.value:#x}+{self.size:#x} t={self.type} sh={self.shndx}>"


class Reloc:
    def __init__(self, offset, typ, sym, addend):
        self.offset, self.type, self.sym, self.addend = offset, typ, sym, addend


class Elf:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = f.read()
        d = self.data
        if d[:4] != b"\x7fELF" or d[4] != 2 or d[5] != 1:
            raise ValueError("not an ELF64 little-endian file: %s" % path)
        (self.e_type, self.e_machine, _ver, self.entry, _phoff, shoff, _flags,
         _ehsize, _phentsize, _phnum, shentsize, shnum, shstrndx) = struct.unpack_from(
            "<HHIQQQIHHHHHH", d, 16)
        self.is_rel = self.e_type == 1
        raw = []
        for i in range(shnum):
            raw.append(struct.unpack_from("<IIQQQQIIQQ", d, shoff + i * shentsize))
        strtab_off = raw[shstrndx][4] if shnum else 0
        self.sections = []
        for i, (nm, typ, flags, addr, off, size, link, info, _al, ent) in enumerate(raw):
            self.sections.append(Section(i, self._cstr(strtab_off + nm), typ, flags,
                                         addr, off, size, link, info, ent))
        self.symbols = []      # from .symtab (or .dynsym if no symtab)
        symtabs = [s for s in self.sections if s.type == SHT_SYMTAB] + \
                  [s for s in self.sections if s.type == SHT_DYNSYM]
        self._symtab_index = {}
        for st in symtabs:
            strs = self.sections[st.link].offset
            n = st.size // 24
            lst = []
            for i in range(n):
                nm, info, _oth, shndx, val, size = struct.unpack_from("<IBBHQQ", d, st.offset + i * 24)
                name = self._cstr(strs + nm)
                typ, bind = info & 0xF, info >> 4
                if typ == STT_SECTION and 0 < shndx < len(self.sections):
                    name = self.sections[shndx].name
                sym = Symbol(name, val, size, typ, bind, shndx)
                lst.append(sym)
            self._symtab_index[st.idx] = lst
            if st is symtabs[0]:
                self.symbols = lst        # .symtab, or .dynsym when stripped
        # relocations grouped by target section index
        self.relocs = {}
        for s in self.sections:
            if s.type != SHT_RELA:
                continue
            syms = self._symtab_index.get(s.link, self.symbols)
            lst = []
            for i in range(s.size // 24):
                off, info, add = struct.unpack_from("<QQq", d, s.offset + i * 24)
                si, typ = info >> 32, info & 0xFFFFFFFF
                sym = syms[si] if si < len(syms) else None
                lst.append(Reloc(off, typ, sym, add))
            self.relocs.setdefault(s.info, []).extend(lst)

    def _cstr(self, off):
        end = self.data.index(b"\0", off)
        return self.data[off:end].decode("latin-1")

    # ---- queries -------------------------------------------------------
    def section(self, name):
        for s in self.sections:
            if s.name == name:
                return s
        return None

    def section_of_addr(self, addr):
        """Section containing a virtual address (executables / shared objects)."""
        for s in self.sections:
            if s.flags & 2 and s.addr <= addr < s.addr + max(s.size, 1) and s.addr:
                return s
        return None

    def read(self, sec, off, n):
        if sec.type == SHT_NOBITS:
            return None
        if off < 0 or off + n > sec.size:
            return None
        return self.data[sec.offset + off: sec.offset + off + n]

    def cstring(self, sec, off, maxlen=4096):
        if sec is None or sec.type == SHT_NOBITS or not (0 <= off < sec.size):
            return None
        raw = self.data[sec.offset + off: sec.offset + min(sec.size, off + maxlen)]
        end = raw.find(b"\0")
        if end < 0:
            return None
        s = raw[:end]
        try:
            txt = s.decode("utf-8")
        except UnicodeDecodeError:
            return None
        if any((ord(c) < 32 and c not in "\n\t\r\x1b") for c in txt):
            return None
        return txt

    def data_symbols(self):
        return [s for s in self.symbols if s.type in (STT_OBJECT, STT_NOTYPE) and s.name
                and 0 < s.shndx < len(self.sections) and s.type != STT_SECTION]

    def symbol_at(self, shndx, off):
        """Best data/function symbol covering section offset (rel) or address (exe)."""
        best = None
        for s in self.symbols:
            if s.type == STT_SECTION or not s.name or s.shndx != shndx:
                continue
            if s.value <= off < s.value + max(s.size, 1):
                if best is None or s.size < best.size:
                    best = s
        return best
