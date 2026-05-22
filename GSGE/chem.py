
NEGATIVE_RING_SYMBOL = "->"
POP_SYMBOL = "pop"
RINGS_SYMBOLS = ["Ring1", "Ring2", "Ring3"]
BRANCH_SYMBOLS = ["Branch", "=Branch", "#Branch"]
ATTACHEMENT_SYMBOLS = [f':{i}' for i in range(15)]
SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

_ELEMENTS_BOND_COUNTS = {
        'B': [1, 2, 3],
        'C': [1, 2, 3, 4],
        'N': [1, 2, 3],
        'O': [1, 2],
        'F': [1],
        'Si': [1, 2, 3],
        'P': [1, 2, 3, 4, 5],
        'S': [1, 2, 3, 4, 5, 6],
        'Cl': [1],
        'Ge': [1, 2, 3],
        'As': [1, 2, 3, 4],
        'Se': [1, 2, 3, 4, 5],
        'Br': [1],
        'Zr': [1, 2, 3, 4],
        'Sn': [1, 2, 3],
        'Te': [1, 2, 3, 4, 5],
        'I': [1],
        'Hg': [1, 2],
        'Pb': [1, 2, 3],
        }

_ELEMENT_TOKENS = list(_ELEMENTS_BOND_COUNTS.keys())

_GRAMMAR_TOKENS = [NEGATIVE_RING_SYMBOL, POP_SYMBOL] + RINGS_SYMBOLS + BRANCH_SYMBOLS + ATTACHEMENT_SYMBOLS

#Redirect to simple vocab
_REDIRECT_PROCESS_ATOM_CACHE = {
        '[#C+1]':'[C]', 
        '[#C-1]':'[C]', 
        '[#C]'  :'[C]', 
        '[#N+1]':'[N]',
        '[#N]'  :'[N]', 
        '[#O+1]':'[O]', 
        '[#P+1]':'[P]', 
        '[#P-1]':'[P]', 
        '[#P]'  :'[P]', 
        '[#S+1]':'[S]', 
        '[#S-1]':'[S]', 
        '[#S]'  :'[S]', 
        '[=C+1]':'[C]', 
        '[=C-1]':'[C]', 
        '[=C]'  :'[C]', 
        '[=N+1]':'[N]', 
        '[=N-1]':'[N]', 
        '[=N]'  :'[N]', 
        '[=O+1]':'[O]', 
        '[=O]'  :'[O]', 
        '[=P+1]':'[P]', 
        '[=P-1]':'[P]', 
        '[=P]'  :'[P]', 
        '[=S+1]':'[S]', 
        '[=S-1]':'[S]', 
        '[=S]'  :'[S]', 
        '[Br]'  :'[Br]', 
        '[C+1]' :'[C]', 
        '[C-1]' :'[C]', 
        '[C]'   :'[C]', 
        '[Cl]'  :'[Cl]', 
        '[F]'   :'[F]', 
        '[H]'   :'[H]', 
        '[I]'   :'[I]', 
        '[N+1]' :'[N]', 
        '[N-1]' :'[N]', 
        '[N]'   :'[N]', 
        '[O+1]' :'[O]', 
        '[O-1]' :'[O]', 
        '[O]'   :'[O]', 
        '[P+1]' :'[P]', 
        '[P-1]' :'[P]', 
        '[P]'   :'[P]', 
        '[S+1]' :'[S]', 
        '[S-1]' :'[S]', 
        '[S]'   :'[S]'
}

# Remove '[' and ']' from keys and values
_REDIRECT_TOKENS = {k.strip("[]"): v.strip("[]") for k, v in _REDIRECT_PROCESS_ATOM_CACHE.items()}

COMMON_SMALLER_FRAGMENTS = [ #TODO this list can be improved and some of these maybe redundand or invalid
    # Carbonyls, thiocarbonyls, imines
    'O=C(*1)(*1)', 'S=C(*1)(*1)', 'S=C(*1)', 'N=C(*1)(*1)',
    'C=O(*1)', 'C=S(*1)', 'O=S(*1)', 'O=S(*1)(*1)',
    'C=S(*1)', '(*1)C=S(*1)',

    # Alkenes, alkynes, allenes
    'C=C(*1)(*1)', 'C(*1)(*1)=C(*1)(*1)', 'C=C=C(*1)(*1)',
    'C#C(*1)', 'C#N',

    # Azo, azide, nitroso, diazonium, small amines
    'N(*1)(*1)', 'N=O', 'N(*1)=C(*1)(*1)', '*1N=N(*1)',
    '[N-]=[N+]=N(*1)', 'N=[N+]=N(*1)', '*1CN=[N+]=N', '[N+](*1)(*1)',

    # Nitro, phospho, sulfonyl groups
    'O=[N+](*1)(*1)', 'P=O(*1)(*1)', 'O=P(*1)(*1)(*1)', 
    'O=P(*1)(*1)', 'S(=O)(=O)(*1)',

    # Alcohols, ethers, amines, thiols
    'C-O(*1)', 'C-N(*1)', 'C-S(*1)',

    # Methyl, methylene, alkyl linkers
    'C(*1)(*1)', 'CC(*1)(*1)', 'CC(*1)C', 'CC(*1)O', 'CH(*1)(*1)',

    # Halogens
    'Cl(*1)', 'F(*1)', 'Br(*1)', 'I(*1)',

    # Lone heteroatoms
    'O(*1)', 'S(*1)', 'N(*1)',

    # Other
    '[SeH]C(*1)'
]
