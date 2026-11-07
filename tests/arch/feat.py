import re, sys
feat = {
 'pred-exec': r'\? ~?%pred',
 'PLU': r'^\s+(pass|landp|andp)',
 'merge': r'merge[sd]',
 'spec-ld': r'^\s+ld\w*,\d,sm',
 'branch': r'ct %ctpr[12]',
 'rot-%b': r'%d?b\[',
 'loop-hw': r'loop_mode|NOT_LOOP_END',
 'AAU': r'aau|mova|staa|ldaa',
 'call-wbs': r'call .*wbs',
}
for fn in sys.argv[1:]:
    txt=open(fn).read()
    print('=====', fn)
    for f in re.split(r'\n(?=[0-9a-f]+ <)', txt)[1:]:
        name=re.match(r'[0-9a-f]+ <(\w+)>',f).group(1)
        bundles=re.split(r'\n\s+[0-9a-f]+:\s*\n',f)[1:]
        ops=[len([l for l in b.split('\n') if l.strip() and not re.match(r'\s+(nop|ipd|\d|R_)',l) and 'R_E2K' not in l]) for b in bundles]
        got=[k for k,r in feat.items() if re.search(r,f,re.M)]
        print(f'{name:10} ШК={len(bundles):3} max_ops={max(ops,default=0):2}  '+' '.join(got))
