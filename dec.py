import numba
import numpy as np
from pathlib import Path

alphabet = np.arange(ord('a'), ord('z')+1, dtype='u1')
np.set_printoptions(formatter={'int': lambda v: chr(v) if v in alphabet else str(v)})

def as_u1(v: str):
    if not isinstance(v, str):
        return v.copy()
    
    return np.char.array(v, unicode=False).astype(str).encode('ascii').view('u1').copy()

def as_str(arr):
    return ''.join(map(lambda v: chr(v), arr.flatten()))

def pad_a_to_b(a, b):
    return np.tile(a, int(np.ceil(len(b)/len(a))))[:len(b)]

def histogram(arr):
    keys = np.unique(arr)
    counters = np.zeros(len(keys), dtype=np.int64)
    for i, k in enumerate(keys):
        counters[i] = np.count_nonzero(arr == k)

    rank = np.argsort(counters)[::-1]
    return np.stack([keys[rank], counters[rank]])

@numba.njit
def find_repeats(arr, min_len=3, max_len=6):
    visited = []
    repeats = []
    for i in list(range(min_len, max_len+1))[::-1]:
        for src_cursor in range(len(arr)-i):
            if src_cursor in visited:
                continue
            
            src = arr[src_cursor:src_cursor+i]
            for j in range(src_cursor+i, len(arr)-i):
                # print(src, arr[j:j+i])
                if (src == arr[j:j+i]).all():
                    visited.append(src_cursor)
                    repeats.append((src_cursor, j-src_cursor, src))
    return repeats

def advindexing_roll(A, r):
    rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]    
    r[r < 0] += A.shape[1]
    column_indices = column_indices - r[:,np.newaxis]
    return A[rows, column_indices]

def make_shifted_table(prefix='kryptos'):
    def move_to_front(arr, front: str):
        front = list(reversed(front))
        idx_array = []
        while len(front):
            v = ord(front.pop())
            idx, = np.where(arr == v)
            idx_array.append(idx[0])

        for i in range(len(arr)):
            if i not in idx_array:
                idx_array.append(i)
        return idx_array
            
    
    row = alphabet.copy()
    idxs = move_to_front(row, prefix)
    row = row[idxs] # Top row done
    table = np.tile(row, (26, 1))
    table = advindexing_roll(table, -np.arange(len(table)))
    return table

def table_idxs(v, idx_row):
    if idx_row.ndim == 1:
        idx_row = idx_row[:, None]

    v_idxs = np.where(idx_row == v)
    v_idxs = v_idxs[0][np.argsort(v_idxs[1])]
    # assert table[0][v_idxs] == v
    return v_idxs

def inv_idx_row(idx_row):
    return table_idxs(alphabet, idx_row)+ord('a')

def apply_table_idx(v, idx_row, reverse=False):
    v = as_u1(v)
    
    if reverse:
        idx_row = inv_idx_row(idx_row)

    idxs = table_idxs(v, idx_row)
    return idxs + ord('a')
    

def vig_enc(a, b, table):
    a = as_u1(a)
    b = as_u1(b)
    
    # Get vignier index of each letter
    a_idxs = table_idxs(a, table[0])
    b_idxs = table_idxs(b, table[0])

    if len(b_idxs) != len(a_idxs):
        # Pad letters to match largest component
        b_idxs = np.tile(b_idxs, int(np.ceil(len(a_idxs)/len(b_idxs))))[:len(a_idxs)]
        a_idxs = np.tile(a_idxs, int(np.ceil(len(b_idxs)/len(a_idxs))))[:len(b_idxs)]
        # print(f'{b_idxs.shape=}, {a_idxs.shape=}')

    # Final output at intersection of 2 texts
    return table[b_idxs, a_idxs]


def vig_dec(cipher, key, table):
    cipher = as_u1(cipher)
    key = as_u1(key)

    # Pad key to cipher
    key = np.tile(key, int(np.ceil(len(cipher)/len(key))))[:len(cipher)]
    
    k_idxs = table_idxs(key, table[0])
    v_idxs = table_idxs(cipher, table[k_idxs].T) # 2D lookup-table
    # print(k_idxs, v_idxs, key)
    return table[0, v_idxs]

# def vig_dec(cipher, key, table):
#     cipher = as_u1(cipher)
#     key = as_u1(key)
#     # Pad key to cipher
#     key = np.tile(key, int(np.ceil(len(cipher)/len(key))))[:len(cipher)]
#     k_idxs = table_idxs(key, table[0])
#     c_idxs = table_idxs(cipher, table[0])
#     return ((c_idxs - k_idxs) % 26) + ord('a')
    

def extract_invalid(arr):
    arr = as_u1(arr)
    mask = np.logical_or(arr < ord('a'), arr > ord('z'))
    invalid = np.where(mask)
    return arr[~mask], (invalid[0], arr[mask]) # (index, value)

def recursive_splice(arr, index, value):
    arr = as_u1(arr)
    for i, v in zip(index, value):
        arr = np.insert(arr, i, v)
    return arr

def vector_matrix_factors(seqlen):
    shapes = set()
    for i in range(2, seqlen//2):
        if seqlen % i == 0:
            shapes.add((seqlen//i, i))
            shapes.add((i, seqlen//i))
    return shapes

def rot_clockwise(matrix):
    return matrix.T[..., ::-1]

def rot_counterclockwise(matrix):
    return matrix[..., ::-1].T

def skip_transposition(arr, k):
    # This is also a decomposition of k3 rotational transposition
    visited = set()
    out = arr.copy()

    v = k-1
    for i in range(len(arr)):
        if v in visited:
            # Invalid k
            return None

        visited.add(v)
        out[i] = arr[v]
        v += k
        v %= len(arr)
        
    # out[i] = arr[v]
    # print(k, len(visited), visited)
    return out


def transposition_visitor(vec_mat, d=2, mix_f=rot_counterclockwise, post_f=False, stack=None):
    # Dec: mix_f=rot_counterclockwise, post_f=True
    # Enc: mix_f=rot_clockwise, post_f=False

    if stack is None:
        stack = []

    def apply(shape):
        vm = vec_mat.copy()
        if post_f:
            return mix_f(vm.reshape(shape))
        else:
            return mix_f(vm).reshape(shape)
    
    len = np.prod(vec_mat.shape)
    combinations = vector_matrix_factors(len)
    for shape in combinations:
        if d <= 1:
            yield apply(shape), [*stack, shape]
        else:
            yield from transposition_visitor(apply(shape), d=d-1, mix_f=mix_f, post_f=post_f, stack=[*stack, shape])

vig_table = make_shifted_table()
overflow_table = vig_table[:, :4].copy() # w/o center L
print(vig_table)
print(overflow_table)

cipher = vig_enc('secretmessage', 'hidden', vig_table)
print(f'test {cipher=}')
dec_cipher = vig_dec(cipher, 'hidden', vig_table)
print(f'test {dec_cipher=}')

k1 = vig_dec('EMUFPHZLRFAXYUSDJKZLDKRNSHGNFIVJYQTQUXQBQVYUVLLTREVJYQTMKYRDMFD'.lower(), 'palimpsest', vig_table)
print(f'{k1=} {as_str(k1)=}')

k2_org = k2 = 'VFPJUDEEHZWETZYVGWHKKQETGFQJNCEGGWHKK?DQMCPFQZDQMMIAGPFXHQRLGTIMVMZJANQLVKQEDAGDVFRPJUNGEUNAQZGZLECGYUXUEENJTBJLBQCRTBJDFHRRYIZETKZEMVDUFKSJHKFWHKUWQLSZFTIHHDDDUVH?DWKBFUFPWNTDFIYCUQZEREEVLDKFEZMOQQJLTTUGSYQPFEUNLAVIDXFLGGTEZ?FKZBSFDQVGOGIPUFXHHDRKFFHQNTGPUAECNUVPDJMQCLQUMUNEDFQELZZVRRGKFFVOEEXBDMVPNFQXEZLGREDNQFMPNZGLFLPMRJQYALMGNUVPDXVKPDQUMEBEDMHDAFMJGZNUPLGEWJLLAETG'.lower()

print(f'{len(k2)=}')
repeats = find_repeats(as_u1(k2))
print(f'{repeats=}')

k2c, invalid_info = extract_invalid(k2)
k2 = vig_dec(k2c, 'abscissa', vig_table)

# k2c = rot_counterclockwise(k2c.reshape((41, 9))).flatten()
rev_k2 = vig_dec(k2c, k2, vig_table)
print(f'{as_str(rev_k2)=}')

k2 = recursive_splice(k2, *invalid_info)
print(f'{k2=} {as_str(k2)=}')

# it|was|totally|invisible|hows|that|possible|?|they|used|the|earths|magnetic|field|x|the|information|was|gathered|and|transmitted|undergruund|to|an|unknown|location|x|does|langley|know|about|this|?|they|should|its|buried|out|there|somewhere|x|who|knows|the|exact|location|?|only|ww|this|was|his|last|message|x|thirty|eight|degrees|fifty|seven|minutes|six|point|five|seconds|north|seventy|seven|degrees|eight|minutes|forty|four|seconds|west|id|by|rows|
# ID by row(s)


    

k3 = 'ENDYAHROHNLSRHEOCPTEOIBIDYSHNAIACHTNREYULDSLLSLLNOHSNOSMRWXMNETPRNGATIHNRARPESLNNELEBLPIIACAEWMTWNDITEENRAHCTENEUDRETNHAEOETFOLSEDTIWENHAEIOYTEYQHEENCTAYCREIFTBRSPAMHHEWENATAMATEGYEERLBTEEFOASFIOTUETUAEOTOARMAEERTNRTIBSEDDNIAAHTTMSTEWPIEROAGRIEWFEBAECTDDHILCEIHSITEGOEAOSDDRYDLORITRKLMLEHAGTDHARDPNEOHMGFMFEUHEECDMRIPFEIMEHNLSSTTRTVDOHW?'.lower()
k3, invalid_info = extract_invalid(k3)
print(f'{len(k3)=}')
# print(f'{vector_matrix_factors(len(k3))=}')

# Proposed A:
# k3 = rot_counterclockwise(k3.reshape((14, 24)))
# print(k3)
# k3 = rot_counterclockwise(k3.reshape((42, 8)))
# print(k3, as_str(k3))

# Proposed B:
# k3 = skip_transposition(k3, 192) # NOTE including ? at end


for candidate, stack in transposition_visitor(k3, d=2, post_f=True):
    str_candidate = as_str(candidate)
    if 'the' in str_candidate and 'was' in str_candidate:
        print(f'{str_candidate=} {stack=}')
        break


print(f'=== K4 ===')
# slowly|desparatly|slowly|the|remains|of|passage|debris|that|encumbered|the|lower|part|of|the|doorway|was|removed|with|trembling|hands|i|made|a|tiny|breach|in|the|upper|left|hand|corner|and|then|widening|the|hole|a|little|i|inserted|the|candle|and|peered|in|the|hot|air|escaping|from|the|chamber|caused|the|flame|to|flicker|but|presently|details|of|the|room|within|emerged|from|the|mist|x|can|you|see|anything|q|?
# https://archive.org/details/tomboftutankhame01cart/page/n153/mode/2up?q=tiny+breach
# print(f'{k3.shape=} {histogram(k3)=}')
# k3 = recursive_splice(k3, *invalid_info)

def coded_cesear(ct, key):
    # Key entries can be [-26, 26]
    ct = as_u1(ct)
    key = as_u1(key)
    if (key > ord('a')).any():
        key -= ord('a')

    if type(key) != int:
        # Pad
        key = np.tile(key, int(np.ceil(len(ct)/len(key))))[:len(ct)]

    dec = ct + key
    # ring-group
    dec[dec < ord('a')] += 26
    dec[dec > ord('z')] -= 26
    return dec


# Remaining hints (chekov's gun):
# - Mispellings: l->q, o->u, e->a, e->None +-(-5, -6, 4, 4|-4?)
# - "Layer two" (may be due to double rot of k3 rather than k4)
# - "At least 2 ciphers" on the entire text - author (Vig + Transposition)
# ? "PS ITS AS SIMPLE AS ABC"

# K2 for comparison
# k4           = 'EMUFPHZLRFAXYUSDJKZLDKRNSHGNFIVJYQTQUXQBQVYUVLLTREVJYQTMKYRDMFD'.lower()
# k4_hints_str = '???????subtleshadingand????????????????????????????????iqlusion'

k4           = 'obkruoxoghulbsolifbbwflrvqqprngkssotwtqsjqssekzzwatjkludiawinfbnypvttmzfpkwgdkzxtjcdigkuhuauekcar'
k4_hints_str = '?????????????????????eastnortheast?????????????????????????????berlinclock???????????????????????'
# k4_hints_str = '?????????????????????????northeast?????????????????????????????berlinclock???????????????????????'

k4_hints      = as_u1(k4_hints_str)
k4_hints_mask = k4_hints != ord('?')
print(f'Got {k4_hints_mask.sum()} known out of {len(k4)}')

for token in ['e', 't', 'c', 'a', 'o', 's', 'l', 'r', 'n']:
    e_mask = k4_hints == ord(token)
    print(token, np.where(e_mask), np.diff(np.where(e_mask)), as_u1(k4)[e_mask], as_u1(k4)[e_mask].astype(np.int64) - ord('a') - table_idxs(k4_hints[e_mask], vig_table[0]).astype(np.int64))
# NOTE: the cipher seems to be cesear-esque but with a potentially indipendent base/direction/mod for each token
# p=position in string
# c=unknown constant
# l, c seems to be +1*p+c (c=-4 and c=-7)
# a & s seems to be (p mod 8)*25+c (for +9 -> -1)

    
# k4_dist      = table_idxs(as_u1(k4), vig_table[0])[k4_hints_mask].astype(np.int64) - table_idxs(as_u1(k4_hints_str.replace('?', 'p')), vig_table[0])[k4_hints_mask].astype(np.int64)
# k4_dist      = as_u1(k4)[k4_hints_mask].astype(np.int64) - as_u1(k4_hints_str)[k4_hints_mask].astype(np.int64)
# print(f'{k4_dist[:13]=} {k4_dist[13:]=}')
rev_k4 = vig_dec(k4, k4_hints_str.replace('?', 'p'), vig_table) # Get key from cipher+known
# rev_k4 = vig_dec(k4, rev_k4, vig_table) # Back to k4_hint
# rev_k4 here doesn't really seem to have a pattern :thinking:
# print(f'{rev_k4[k4_hints_mask][:13]=} {rev_k4[k4_hints_mask][13:]=}')
# rev_k4 = vig_dec(rev_k4, k4, vig_table) # Assume 2 layer, get key from cipher+known
# # Seems like a simple replacement after this point
# print(f'{rev_k4[k4_hints_mask][:13]=} {rev_k4[k4_hints_mask][13:]=}')
# print(f'rev_k4_dist:', rev_k4[k4_hints_mask].astype(np.int64) - as_u1(k4_hints_str)[k4_hints_mask].astype(np.int64))

# print(histogram(as_u1(k4)))
# print('\nAttempting:')

# k4d = vig_dec(k4_key_hat.replace('?', 'p'), k4, vig_table) # Get key from cipher+known
# print(f'{k4d=} {k4d[k4_hints_mask]=}')

# TODO: post/pre-transposition by 24 (should tile over 97)
print(k4_hints[k4_hints_mask])
# key = np.array([0,0,5,6,1,19,23,24,3,2,20,1,7,0,1,16,0,0,0])
# print(f'{len(key)=}')
# for i in range(26):
#     key[14] = i

#     k4t = apply_table_idx(k4, vig_table[0], reverse=True)
#     k4t = coded_cesear(k4t, key)[k4_hints_mask]
#     k4t = apply_table_idx(k4t, vig_table[0])
#     print('cc', k4t, i)        

shift_k = 60
def bruteforce_key(k4, key_len):
    key = np.zeros(key_len)
    for i in range(key_len):
        modification_entries = np.zeros(len(k4), dtype=np.bool_)
        modification_entries[i::key_len] = True
        modification_entries = skip_transposition(modification_entries, k=shift_k) # Comment for k2 solver
        to_check = modification_entries[k4_hints_mask]

        for j in range(26):
            key[i] = j
            k4t = apply_table_idx(k4, vig_table[0], reverse=False)
            k4t = coded_cesear(k4t, key)
            k4t = skip_transposition(k4t, k=shift_k) # Comment for k2 solver
            k4t = k4t[k4_hints_mask]
            k4t = apply_table_idx(k4t, vig_table[0], reverse=True)

            if (diff := (k4t[to_check] != k4_hints[k4_hints_mask][to_check]).sum()) != to_check.sum():
                # print(f'{key=}')
                # print(f'dec={as_str(k4t)}')
                # print(f'mod={as_str(to_check+ord("a"))}')
                # print(f'org={as_str(k4_hints[k4_hints_mask])}')
                # print('Found potential new key entry')
                if diff != 0:
                    print('But it conflicts, this key length is invalid')
                    return
                break
        else:
            key[i] = 26
    return key


# E DIGETAL E E E
# INTERPRETATIT

# T IS YOUR
# POSITION E

# E E VIRTUALLY E
# E E E E E E INVISIBLE

# E E SHADOW E E
# FORCES E E E E E

# LUCID E E E
# MEMORY E

# RQ

# SOS

# K4 seems to be prime length so all skip_transpositions are possible
# k4u = as_u1(k4)
# for k in range(2, len(k4u)):
#     if skip_transposition(k4u, k=k) is not None:
#         print(f'Skip transpos {k=} possible')

# Vig; up to key length 38 is not possible
# (skip/rot) Transpose incl reverse -> Vig/Cesear; is not possible decoder for key lengths <30
# Vig -> Tranpose; also not possible for key lengths <32

# k4u = as_u1(k4)
# for k in range(2, 97):
#     k4 = skip_transposition(k4u, k=k)
    # NOTE: up to a key-length of 38 it doesn't seem to be vig (at least given the known table)
for key_len in range(2, 38):
    v = bruteforce_key(k4, key_len)
    if v is not None:
        v = v.astype(np.int64)
        pv = pad_a_to_b(v, k4)
        print(f'========== {key_len=} ===========')
        print(f'{v=}')
        k4t = apply_table_idx(k4, vig_table[0], reverse=False)
        k4t = coded_cesear(k4t, v)
        unk = k4t[pv == 26]
        k4t[pv == 26] = ord('?')
        
        # Do tranpositions here
        k4t = skip_transposition(k4t, k=shift_k) # Comment for k2 solver
        
        unk_mask = k4t == ord('?')
        k4t[unk_mask] = unk
        k4t = apply_table_idx(k4t, vig_table[0], reverse=True)
        print(f'{as_str(k4t)=} {len(k4t)=}')
        k4t[unk_mask] = ord('?')
        print(f'{as_str(k4t)}')
        if key_len < 20:
            print("WHAT????????????")
        # break

        

# print(overflow_table, overflow_table.shape)
# key = overflow_table[2:14] # + 'L'
# # key = as_u1('wonderfulthings')
# print(key, len(key))

# k4h = vig_dec(k4, key.flatten(), vig_table)
# print(k4h)
# print(histogram(k4h))

# NOTE: k4 needs at least 1 substitution as there aren't enough 'e's to match the hints
# > It could also be that the ordinal-first hint (east) is invalid since it's a second-hand account
def k4_transposition_mismatch(arr):
    # Check if a transposition could be done as a solution
    k4_counts = dict(histogram(k4_hints[k4_hints_mask]).T)
    counts = dict(histogram(arr).T)
    # print(f'{k4_counts=} {counts=}')
    missing = {}
    
    for k, c in k4_counts.items():
        if counts.get(k, 0) < c:
            missing[chr(k)] = c - counts.get(k, 0)

    return missing

def plausible_histogram(hist):
    # Check ranks of most/least common characters
    return (ord('e') in hist[:3]) and (ord('a') in hist[1:5]) and (ord('t') in hist[:4]) and (ord('q') in hist[-4:]) and (ord('j') in hist[-6:]) and (ord('z') in hist[-6:])

def k4_match_cond(arr):
    return arr[k4_hints_mask] == k4_hints[k4_hints_mask]

# for candidate, stack in transposition_visitor(as_u1(k4), d=2, post_f=True):
#     candidate = candidate.flatten()
#     repeats = find_repeats(candidate)
#     print(f'{repeats=} {stack=}')

#     rev_k4 = vig_dec(candidate, k4_hints_str.replace('?', 'p'), vig_table)
#     print(f'{rev_k4[k4_hints_mask]=}')
#     transpose = k4_transposition_mismatch(rev_k4)
#     hist = histogram(rev_k4)
#     pl_hist = plausible_histogram(rev_k4) 
#     print(f'{pl_hist=} {transpose=} {repeats=} {stack=}')
    
    # if k4_match_cond(candidate.flatten()).sum() > 10:
    #     print('Found')
    #     print(as_str(candidate.flatten()))

def hypothetical_k4():
    sample_str = Path('./sample.txt').read_text()
    sample_str = sample_str.replace('\n', ' ').replace('  ', ' ').lower()
    # Create dummy k4 from wordlist + known words to do histgram differential analysis
    words = sample_str.split(' ')
    words = np.random.choice(words, size=30)
    sample, _ = extract_invalid(''.join(words))
    k4_hat = k4_hints.copy()
    k4_hat[~k4_hints_mask] = sample[:len(k4_hat)][~k4_hints_mask]
    return k4_hat

# k4_hat = hypothetical_k4()
# print(k4_hat)
# print(f'{histogram(k4_hat)=}')




# print(f'Plausible histogram {plausible_histogram(k4h)=}')
# print(f'Could transpose {k4_transposition_mismatch(k4h)=}')
# print(f'Could transpose {k4_transposition_mismatch(k4_hat)=}')
# print(f'Could transpose {k4_transposition_mismatch(as_u1(k4))=}')
    
# TODO: for K4 try coded-ceasar cipher by the column-wise alphabet offset of the mispellings    
    

# TODO: add repetition detection, and execute after the transposition_explorer as in k3
