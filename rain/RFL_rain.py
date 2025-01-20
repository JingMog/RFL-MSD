import sys
sys.path.insert(0, "./RFL_")
from RFL import cs_main, chemstem2chemfig

if __name__ == '__main__':
    str1 = 'H B r + \chemfig { ?[a] -[:330] -[:30] -[:90] ( =[:45] ?[b] ( -[:0] ?[c] ( -:[:0] \circle ) ( -[:60] ( -[:0] -[:300] -[:240] -[:180] ?[c,{-}] ) -[:135] \Chemabove { N } { H } -[:210] ?[b,{-}] =[:150] O ) ) ) -[:165] \Chemabove { N } { H } ?[a,{-}] }'
    success, cs_string, branch_info, ring_branch_info, cond_data = cs_main(str1, is_show=True)
    print(success)
    print(cs_string)
    print(ring_branch_info)





