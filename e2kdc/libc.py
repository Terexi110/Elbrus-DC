"""Prototypes of common C library functions.

Used for calls to imported functions (`printf@plt` in an executable, an
undefined symbol in an object file):

* the number of arguments: fixed for ordinary functions; for printf-like
  functions with a literal format string it is counted from the format
  (lcc also spills variadic arguments to a stack area, which otherwise shows
  up as extra arguments);
* the types of the arguments: a variable passed where a `char *` is expected
  (or matched by `%s`) is a pointer to char;
* the declaration printed in the header.

Each entry: name -> (C declaration, return kind, [param kinds], index of the
format argument or None).  Kinds: "int", "long", "ulong", "double", "void",
"ptr" (void *), "str" (char *).
"""
import re

P = {}


def _d(decl, ret, params, fmt=None):
    name = re.search(r"(\w+)\s*\(", decl).group(1)
    P[name] = (decl, ret, params, fmt)


_d("int printf(const char *, ...);", "int", ["str"], 0)
_d("int fprintf(void *, const char *, ...);", "int", ["ptr", "str"], 1)
_d("int sprintf(char *, const char *, ...);", "int", ["str", "str"], 1)
_d("int snprintf(char *, unsigned long, const char *, ...);", "int", ["str", "ulong", "str"], 2)
_d("int dprintf(int, const char *, ...);", "int", ["int", "str"], 1)
_d("int scanf(const char *, ...);", "int", ["str"], 0)
_d("int __isoc99_scanf(const char *, ...);", "int", ["str"], 0)
_d("int sscanf(const char *, const char *, ...);", "int", ["str", "str"], 1)
_d("int __isoc99_sscanf(const char *, const char *, ...);", "int", ["str", "str"], 1)
_d("int fscanf(void *, const char *, ...);", "int", ["ptr", "str"], 1)
_d("int __isoc99_fscanf(void *, const char *, ...);", "int", ["ptr", "str"], 1)
_d("int puts(const char *);", "int", ["str"])
_d("int putchar(int);", "int", ["int"])
_d("int getchar(void);", "int", [])
_d("int fputs(const char *, void *);", "int", ["str", "ptr"])
_d("int fputc(int, void *);", "int", ["int", "ptr"])
_d("char *fgets(char *, int, void *);", "str", ["str", "int", "ptr"])
_d("void *fopen(const char *, const char *);", "ptr", ["str", "str"])
_d("int fclose(void *);", "int", ["ptr"])
_d("unsigned long fread(void *, unsigned long, unsigned long, void *);", "ulong", ["ptr", "ulong", "ulong", "ptr"])
_d("unsigned long fwrite(const void *, unsigned long, unsigned long, void *);", "ulong",
   ["ptr", "ulong", "ulong", "ptr"])
_d("int fflush(void *);", "int", ["ptr"])
_d("unsigned long strlen(const char *);", "ulong", ["str"])
_d("int strcmp(const char *, const char *);", "int", ["str", "str"])
_d("int strncmp(const char *, const char *, unsigned long);", "int", ["str", "str", "ulong"])
_d("char *strcpy(char *, const char *);", "str", ["str", "str"])
_d("char *strncpy(char *, const char *, unsigned long);", "str", ["str", "str", "ulong"])
_d("char *strcat(char *, const char *);", "str", ["str", "str"])
_d("char *strchr(const char *, int);", "str", ["str", "int"])
_d("char *strrchr(const char *, int);", "str", ["str", "int"])
_d("char *strstr(const char *, const char *);", "str", ["str", "str"])
_d("char *strdup(const char *);", "str", ["str"])
_d("void *memcpy(void *, const void *, unsigned long);", "ptr", ["ptr", "ptr", "ulong"])
_d("void *memmove(void *, const void *, unsigned long);", "ptr", ["ptr", "ptr", "ulong"])
_d("void *memset(void *, int, unsigned long);", "ptr", ["ptr", "int", "ulong"])
_d("int memcmp(const void *, const void *, unsigned long);", "int", ["ptr", "ptr", "ulong"])
_d("void *malloc(unsigned long);", "ptr", ["ulong"])
_d("void *calloc(unsigned long, unsigned long);", "ptr", ["ulong", "ulong"])
_d("void *realloc(void *, unsigned long);", "ptr", ["ptr", "ulong"])
_d("void free(void *);", "void", ["ptr"])
_d("void exit(int);", "void", ["int"])
_d("void abort(void);", "void", [])
_d("int atoi(const char *);", "int", ["str"])
_d("long atol(const char *);", "long", ["str"])
_d("long strtol(const char *, char **, int);", "long", ["str", "ptr", "int"])
_d("unsigned long strtoul(const char *, char **, int);", "ulong", ["str", "ptr", "int"])
_d("double strtod(const char *, char **);", "double", ["str", "ptr"])
_d("int abs(int);", "int", ["int"])
_d("int rand(void);", "int", [])
_d("void srand(unsigned int);", "void", ["int"])
_d("long time(long *);", "long", ["ptr"])
_d("char *getenv(const char *);", "str", ["str"])
_d("int open(const char *, int, ...);", "int", ["str", "int"])
_d("int close(int);", "int", ["int"])
_d("long read(int, void *, unsigned long);", "long", ["int", "ptr", "ulong"])
_d("long write(int, const void *, unsigned long);", "long", ["int", "ptr", "ulong"])
_d("void qsort(void *, unsigned long, unsigned long, int (*)(const void *, const void *));", "void",
   ["ptr", "ulong", "ulong", "ptr"])


# glibc: messages, errno, numbers, time, files (the utilities in the lcc sysroot)
_d("void error(int, int, const char *, ...);", "void", ["int", "int", "str"], 2)
_d("void error_at_line(int, int, const char *, unsigned int, const char *, ...);", "void",
   ["int", "int", "str", "int", "str"], 4)
_d("char *dcgettext(const char *, const char *, int);", "str", ["str", "str", "int"])
_d("char *__dcgettext(const char *, const char *, int);", "str", ["str", "str", "int"])
_d("char *dgettext(const char *, const char *);", "str", ["str", "str"])
_d("char *gettext(const char *);", "str", ["str"])
_d("char *dcngettext(const char *, const char *, const char *, unsigned long, int);", "str",
   ["str", "str", "str", "ulong", "int"])
_d("char *textdomain(const char *);", "str", ["str"])
_d("char *bindtextdomain(const char *, const char *);", "str", ["str", "str"])
_d("char *setlocale(int, const char *);", "str", ["int", "str"])
_d("int *__errno_location(void);", "ptr", [])
_d("char *strerror(int);", "str", ["int"])
_d("long __isoc23_strtol(const char *, char **, int);", "long", ["str", "ptr", "int"])
_d("unsigned long __isoc23_strtoul(const char *, char **, int);", "ulong", ["str", "ptr", "int"])
_d("long __isoc23_strtoimax(const char *, char **, int);", "long", ["str", "ptr", "int"])
_d("long long __isoc23_strtoll(const char *, char **, int);", "long", ["str", "ptr", "int"])
_d("unsigned long long __isoc23_strtoull(const char *, char **, int);", "ulong", ["str", "ptr", "int"])
_d("void *localtime_r(const long *, void *);", "ptr", ["ptr", "ptr"])
_d("void *gmtime_r(const long *, void *);", "ptr", ["ptr", "ptr"])
_d("void *localtime(const long *);", "ptr", ["ptr"])
_d("void *gmtime(const long *);", "ptr", ["ptr"])
_d("unsigned long strftime(char *, unsigned long, const char *, const void *);", "ulong",
   ["str", "ulong", "str", "ptr"])
_d("long pread64(int, void *, unsigned long, long);", "long", ["int", "ptr", "ulong", "long"])
_d("long pwrite64(int, const void *, unsigned long, long);", "long", ["int", "ptr", "ulong", "long"])
_d("int open64(const char *, int, ...);", "int", ["str", "int"])
_d("int openat64(int, const char *, int, ...);", "int", ["int", "str", "int"])
_d("long readlinkat(int, const char *, char *, unsigned long);", "long", ["int", "str", "str", "ulong"])
_d("void *fdopendir(int);", "ptr", ["int"])
_d("void *readdir64(void *);", "ptr", ["ptr"])
_d("int closedir(void *);", "int", ["ptr"])
_d("int fputs_unlocked(const char *, void *);", "int", ["str", "ptr"])
_d("int putc(int, void *);", "int", ["int", "ptr"])
_d("int __overflow(void *, int);", "int", ["ptr", "int"])
_d("void *memchr(const void *, int, unsigned long);", "ptr", ["ptr", "int", "ulong"])
_d("char *stpcpy(char *, const char *);", "str", ["str", "str"])
_d("unsigned long strspn(const char *, const char *);", "ulong", ["str", "str"])
_d("unsigned long strcspn(const char *, const char *);", "ulong", ["str", "str"])
_d("char *strndup(const char *, unsigned long);", "str", ["str", "ulong"])
_d("int fileno(void *);", "int", ["ptr"])
_d("int ferror(void *);", "int", ["ptr"])
_d("int feof(void *);", "int", ["ptr"])
_d("void *fdopen(int, const char *);", "ptr", ["int", "str"])
_d("int fcntl(int, int, ...);", "int", ["int", "int"])
_d("int ioctl(int, unsigned long, ...);", "int", ["int", "ulong"])
_d("void *mmap(void *, unsigned long, int, int, int, long);", "ptr", ["ptr", "ulong", "int", "int", "int", "long"])
_d("int munmap(void *, unsigned long);", "int", ["ptr", "ulong"])
_d("int unlink(const char *);", "int", ["str"])
_d("int rename(const char *, const char *);", "int", ["str", "str"])
_d("int access(const char *, int);", "int", ["str", "int"])
_d("int iconv_close(void *);", "int", ["ptr"])
_d("unsigned long iconv(void *, char **, unsigned long *, char **, unsigned long *);", "ulong",
   ["ptr", "ptr", "ptr", "ptr", "ptr"])
_d("void *iconv_open(const char *, const char *);", "ptr", ["str", "str"])
_d("long ptrace(int, ...);", "long", ["int"])
_d("int waitpid(int, int *, int);", "int", ["int", "ptr", "int"])
_d("int kill(int, int);", "int", ["int", "int"])
_d("int getopt_long(int, char *const *, const char *, const void *, int *);", "int",
   ["int", "ptr", "str", "ptr", "ptr"])
_d("int argp_parse(const void *, int, char **, unsigned int, int *, void *);", "int",
   ["ptr", "int", "ptr", "int", "ptr", "ptr"])
_d("void argp_help(const void *, void *, unsigned int, char *);", "void", ["ptr", "ptr", "int", "str"])
_d("void argp_error(const void *, const char *, ...);", "void", ["ptr", "str"], 1)
_d("int fflush_unlocked(void *);", "int", ["ptr"])
_d("int __libc_start_main(void *, int, char **, void *, void *, void *, void *);", "int",
   ["ptr", "int", "ptr", "ptr", "ptr", "ptr", "ptr"])


def lookup(name):
    return P.get(name)


def format_args(fmt):
    """number of arguments consumed by a printf/scanf format, or None"""
    n = 0
    kinds = []
    i = 0
    while i < len(fmt):
        if fmt[i] != "%":
            i += 1
            continue
        i += 1
        if i < len(fmt) and fmt[i] == "%":
            i += 1
            continue
        m = re.match(r"[-+ #0']*(\*|\d+)?(?:\.(\*|\d+))?(hh|h|ll|l|L|q|j|z|t)?([diouxXeEfFgGaAcspn\[])", fmt[i:])
        if not m:
            return None
        for w in (m.group(1), m.group(2)):
            if w == "*":
                n += 1
                kinds.append("int")
        n += 1
        conv = m.group(4)
        kinds.append("str" if conv == "s" else "ptr" if conv in "pn[" else "num")
        i += m.end()
        if conv == "[":
            j = fmt.find("]", i + 1)
            i = j + 1 if j >= 0 else len(fmt)
    return n, kinds
