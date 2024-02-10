from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
# import jax
# from functools import partial


NA_VAL = -9999
HIDDEN_VAL = -20000
MASK_VAL = -10000


def encode(ar, ks, vs):
    """ replace values in ar with values in vs at indices in ks 
        (mapping from vs to ks)
    """
    return vs[np.searchsorted(ks, ar)]

def decode(ar, ks, vs):
    return encode(ar, vs, ks)

def is_special_val(x):
    return np.isin(x, np.array([MASK_VAL, HIDDEN_VAL, NA_VAL])).any()

def expand_special_val(x, n_tokens):
    return np.tile(x, (n_tokens,))

def split_int(x, n_tokens, tok_len, prepend_sign_token=False):

    if prepend_sign_token:
        sign = np.sign(x)
        # only allow pos or neg sign, counting zero as negative (-0)
        sign = np.where(sign == 0, -1, sign)
    
    x = np.abs(x)
    base = 10
    n_digits = n_tokens * tok_len
    div_exp = np.flip(
        np.arange(0, n_digits, tok_len))
    splits = (x // (base ** div_exp)) % (10**tok_len)
    if prepend_sign_token:
        splits = np.hstack([sign, splits])
    return splits

def combine_int(x, tok_len, sign=1):
    base = 10
    n_digits = np.expand_dims(x, axis=0).shape[-1] * tok_len
    exp = np.flip(
        np.arange(0, n_digits, tok_len))
    return sign * np.sum(x * (base ** exp), axis=-1)

def split_field(x, n_tokens, tok_len, prepend_sign_token=False):
    total_tokens = n_tokens + int(prepend_sign_token)
    if is_special_val(x):
        return expand_special_val(x, total_tokens)
    else:
        return split_int(x, n_tokens, tok_len, prepend_sign_token)
    # return jax.lax.cond(
    #     is_special_val(x),
    #     lambda arg: expand_special_val(arg, total_tokens),
    #     lambda arg: split_int(arg, n_tokens, tok_len, prepend_sign_token),
    #     x)

def combine_field(
        x: np.array,
        tok_len: int,
        sign: np.array = np.array(1)
    ):
    if is_special_val(np.concatenate((sign.flatten(), x.flatten()))):
        return NA_VAL
    else:
        return combine_int(x, tok_len, sign)
    # return jax.lax.cond(
    #     is_special_val(np.concatenate((sign.flatten(), x.flatten()))),
    #     lambda arg: NA_VAL,
    #     lambda arg: combine_int(arg, tok_len, sign),
    #     x)

# event_type	direction	price	size	delta_t	time_s	time_ns	price_ref	size_ref	time_s_ref	time_ns_ref
# TODO: REIMPLEMENT
def encode_msg(
        msg: np.array,
        encoding: Dict[str, Tuple[np.array, np.array]],
    ) -> np.array:
    event_type = encode(msg[1], *encoding['event_type'])
    
    direction = encode(msg[2], *encoding['direction'])
    # NOTE: leave out price_abs in msg[3]
    price = split_field(msg[4], 1, 3, True)
    # CAVE: temporary fix to switch tokens for + and - sign
    price_sign = encode(price[0], *encoding['sign'])
    price = encode(price[1], *encoding['price'])
    
    size = encode(msg[5], *encoding['size'])
    
    time_comb = encode_time(
        time_s = msg[8], 
        time_ns = msg[9],
        encoding = encoding,
        delta_t_s = msg[6],
        delta_t_ns = msg[7],
    )

    price_ref = split_field(msg[10], 1, 3, True)
    # CAVE: temporary fix to switch tokens for + and - sign
    price_ref_sign = encode(price_ref[0], *encoding['sign'])
    price_ref = encode(price_ref[1], *encoding['price'])

    size_ref = encode(msg[11], *encoding['size'])
    time_ref_comb = encode_time(
        time_s = msg[12], 
        time_ns = msg[13],
        encoding = encoding
    )

    out = [
        event_type, direction, price_sign, price, size, time_comb, # delta_t, time_s, time_ns,
        price_ref_sign, price_ref, size_ref, time_ref_comb]
    return np.hstack(out) # time_s_ref, time_ns_ref])

def encode_msgs(msgs, encoding):
    return np.array([encode_msg(msg, encoding) for msg in msgs])

def encode_time(
        time_s: np.array,
        time_ns: np.array,
        encoding: Dict[str, Tuple[np.array, np.array]],
        delta_t_s: Optional[np.array] = None,
        delta_t_ns: Optional[np.array] = None,
    ) -> np.array:
    # convert seconds after midnight to seconds after exchange open
    # --> 9.5 * 3600 = 34200
    time_s = split_field(time_s, 2, 3, False)
    time_ns = split_field(time_ns, 3, 3, False)
    if delta_t_s is None and delta_t_ns is None:
        time_comb = np.hstack([time_s, time_ns])
    else:
        delta_t_ns = split_field(delta_t_ns, 3, 3, False)
        time_comb = np.hstack([delta_t_s, delta_t_ns, time_s, time_ns])
    time_comb = encode(time_comb, *encoding['time'])
    return time_comb


# TODO: REIMPLEMENT
def decode_msg(msg_enc, encoding):
    # TODO: check if fields with same decoder can be combined into one decode call

    event_type = decode(msg_enc[0], *encoding['event_type'])
    
    direction = decode(msg_enc[1], *encoding['direction'])

    price_sign =  decode(msg_enc[2], *encoding['sign'])
    price = decode(msg_enc[3], *encoding['price'])
    price = combine_field(price, 3, price_sign)

    size = decode(msg_enc[4], *encoding['size'])

    delta_t_s, delta_t_ns, time_s, time_ns = decode_time(msg_enc[5:14], encoding)

    price_ref_sign = decode(msg_enc[14], *encoding['sign'])
    price_ref = decode(msg_enc[15], *encoding['price'])
    price_ref = combine_field(price_ref, 3, price_ref_sign)

    size_ref = decode(msg_enc[16], *encoding['size'])
    time_s_ref, time_ns_ref = decode_time(msg_enc[17:22], encoding)

    # order ID is not encoded, so it's set to NA
    # same for price_abs
    return np.hstack([ NA_VAL,
        event_type, direction, NA_VAL, price, size, delta_t_s, delta_t_ns, time_s, time_ns,
        price_ref, size_ref, time_s_ref, time_ns_ref])

def decode_msgs(msgs, encoding):
    return np.array([decode_msg(msg, encoding) for msg in msgs])

def decode_time(time_toks, encoding):
    if time_toks.shape[0] == 0:
        return NA_VAL, NA_VAL
    time = decode(time_toks, *encoding['time'])
    # delta_t and time given
    if time.shape[0] == 9:
        delta_t_s = time[0]
        delta_t_ns = combine_field(time[1:4], 3)
        time_s = combine_field(time[4:6], 3) #+ 34200
        time_ns = combine_field(time[6:], 3)

        return delta_t_s, delta_t_ns, time_s, time_ns
    # only time given
    elif time.shape[0] == 5:
        # convert time_s to seconds after midnight
        time_s = combine_field(time[:2], 3) #+ 34200
        time_ns = combine_field(time[2:], 3)

        return time_s, time_ns

# TODO: REIMPLEMENT
def repr_raw_msg(msg):
    field_names = ['OID', 'event_type', 'direction', 'price_abs', 'price', 'size',
                   'delta_t_s', 'delta_t_ns', 'time_s', 'time_ns',
                   'p_ref', 'size_ref', 'time_s_ref', 'time_ns_ref']
    out = ''
    for name, val in zip(field_names, msg):
        # TODO: format spacing
        out += name + ":\t" + str(val) + "\n"
    return out

# TODO: REIMPLEMENT
class Vocab:

    MASK_TOK = 0
    HIDDEN_TOK = 1
    NA_TOK = 2

    def __init__(self) -> None:
        self.counter = 3  # 0: MSK, 1: HID, 2: NAN
        self.ENCODING = {}
        self.DECODING = {}
        self.DECODING_GLOBAL = {}
        self.TOKEN_DELIM_IDX = {}

        self._add_field('time', range(1000), [3,6,9,12])
        self._add_field('event_type', range(1,5), None)
        self._add_field('size', range(10000), [])
        self._add_field('price', range(1000), [1])
        self._add_field('sign', [-1, 1], None)
        self._add_field('direction', [0, 1], None)

    def __len__(self):
        return self.counter

    def _add_field(self, name, values, delim_i=None):
        enc = [(MASK_VAL, Vocab.MASK_TOK), (HIDDEN_VAL, Vocab.HIDDEN_TOK), (NA_VAL, Vocab.NA_TOK)]
        enc += [(val, self.counter + i) for i, val in enumerate(values)]
        self.counter += len(enc) - 3  # don't count special tokens
        enc = tuple(zip(*enc))
        self.ENCODING[name] = (
            np.array(enc[0], dtype=np.int32),
            np.array(enc[1], dtype=np.int32))

    def _add_special_tokens(self):
        for field, enc in self.ENCODING.items():
            self.ENCODING[field]['MSK'] = Vocab.MASK_TOK
            self.ENCODING[field]['HID'] = Vocab.HIDDEN_TOK
            self.ENCODING[field]['NAN'] = Vocab.NA_TOK

            self.DECODING[field][Vocab.MASK_TOK] = 'MSK'
            self.DECODING[field][Vocab.HIDDEN_TOK] = 'HID'
            self.DECODING[field][Vocab.NA_TOK] = 'NAN'
        self.ENCODING['generic'] = {
            'MSK': Vocab.MASK_TOK,
            'HID': Vocab.HIDDEN_TOK,
            'NAN': Vocab.NA_TOK,
        }
        self.DECODING_GLOBAL[Vocab.MASK_TOK] = ('generic', 'MSK')
        self.DECODING_GLOBAL[Vocab.HIDDEN_TOK] = ('generic', 'HID')
        self.DECODING_GLOBAL[Vocab.NA_TOK] = ('generic', 'NAN')

# TODO: REIMPLEMENT
class Message_Tokenizer:

    FIELDS = (
        'event_type',
        'direction',
        'price',
        'size',
        'delta_t_s',
        'delta_t_ns',
        'time_s',
        'time_ns',
        # reference fields:
        'price_ref',
        'size_ref',
        'time_s_ref',
        'time_ns_ref',
    )
    N_NEW_FIELDS = 8
    N_REF_FIELDS = 4
    # note: list comps only work inside function for class variables
    FIELD_I = (lambda fields=FIELDS:{
        f: i for i, f in enumerate(fields)
    })()
    TOK_LENS = np.array((1, 1, 2, 1, 1, 3, 2, 3, 2, 1, 2, 3))
    TOK_DELIM = np.cumsum(TOK_LENS[:-1])
    MSG_LEN = np.sum(TOK_LENS)
    # encoded message length: total length - length of reference fields
    NEW_MSG_LEN = MSG_LEN - \
        (lambda tl=TOK_LENS, fields=FIELDS: np.sum(tl[i] for i, f in enumerate(fields) if f.endswith('_ref')))()
    # fields in correct message order:
    FIELD_ENC_TYPES = {
        'event_type': 'event_type',
        'direction': 'direction',
        'price': 'price', #'generic',
        'size': 'size', #'generic',
        'delta_t_s': 'time', #'generic',
        'delta_t_ns': 'time',
        'time_s': 'time', #'generic',
        'time_ns': 'time',
        'price_ref': 'price',
        'size_ref': 'size',
        'time_s_ref': 'time',
        'time_ns_ref': 'time',
    }

    @staticmethod
    def get_field_from_idx(idx):
        """ Get the field of a given index (or indices) in a message
        """
        if isinstance(idx, int) or idx.ndim == 0:
            idx = np.array([idx])
        if np.any(idx > Message_Tokenizer.MSG_LEN - 1):
            raise ValueError("Index ({}) must be less than {}".format(idx, Message_Tokenizer.MSG_LEN))
        field_i = np.searchsorted(Message_Tokenizer.TOK_DELIM, idx, side='right')
        return [Message_Tokenizer.FIELDS[i] for i in field_i]
    
    @staticmethod
    def _generate_col_idx_by_encoder():
        """ Generates attribute dictionary col_idx_by_encoder
            with encoder type as key and a list of column (field)
            indices as value. This is used to efficiently decode tokenized
            data. 
        """
        col_idx_by_encoder = {}
        counter = 0
        for n_toks, (col, enc_type) in zip(
            Message_Tokenizer.TOK_LENS,
            Message_Tokenizer.FIELD_ENC_TYPES.items()):
            add_vals = list(range(counter, counter + n_toks))
            try:
                col_idx_by_encoder[enc_type].extend(add_vals)
            except KeyError:
                col_idx_by_encoder[enc_type] = add_vals
            counter += n_toks
        return col_idx_by_encoder

    #col_idx_by_encoder = _generate_col_idx_by_encoder.__func__()()

    def __init__(self) -> None:
        self.col_idx_by_encoder = self._generate_col_idx_by_encoder()
        pass

    def validate(self, toks, vocab):
        """ checks if toks is syntactically AND semantically valid message
            returns triple of (is_valid, error location, error message)
        """
        valid_synt, res = self._validate_syntax(toks, vocab)
        if not valid_synt:
            return False, res, 'syntax error'
        valid_semant, err = self._validate_semantics(res)
        if not valid_semant:
            return False, None, err

    def _validate_syntax(self, toks, vocab):
        try:
            decoded = self.decode_to_str(toks, vocab, error_on_invalid=True)
            return True, decoded
        except ValueError as e:
            return False, e.err_i

    def _validate_semantics(self, decoded):
        ''' checks if decoded message string is semantically correct
            return tuple of (is_valid, error in field, error message)
        '''
        pass

    def invalid_toks_per_msg(self, toks, vocab):
        return (self.decode_to_str(toks, vocab) == '').sum(axis=-1)
    
    def invalid_toks_per_seq(self, toks, vocab):
        return self.invalid_toks_per_msg(toks, vocab).sum(axis=-1)

    def preproc(self, m, b, allowed_event_types=['A','E','C','D','R']):
        # TYPE
        # filter out only allowed event types ...
        m = m.loc[m.type.isin(allowed_event_types)].copy()
        # ... and corresponding book changes
        b = b.loc[m.index]

        # TIME
        # DELTA_T: time since previous order --> 4 tokens of length 3
        m.insert(
            loc=1,
            column='delta_t_ns',
            value=m['time'].diff().fillna(0)
        )
        m.insert(
            loc=1,
            column='delta_t_s',
            value=m.delta_t_ns.astype(int)
        )
        m.delta_t_ns = ((m.delta_t_ns % 1) * 1000000000).astype(int)

        m.insert(0, 'time_s', m.time.astype(int))
        m.rename(columns={'time': 'time_ns'}, inplace=True)
        m.time_ns = ((m.time_ns % 1) * 1000000000).astype(int)
        
        # SIZE
        m.loc[m['size'] > 9999, 'size'] = 9999
        m.loc[m['oldSize'] > 9999, 'oldSize'] = 9999
        m['size'] = m['size'].astype(int)

        # PRICE
        m['price_abs'] = m.price  # keep absolute price for later (simulator)
        # mid-price reference, rounded down to nearest tick_size
        tick_size = 1
        p_ref = (((b.iloc[:, 1] * 100) + (b.iloc[:, 3] * 100)) / 2).shift()#.round(-2).astype(int).shift()
        p_ref = (p_ref // tick_size) * tick_size
        # --> 1999 price levels // ...00 since tick size is 100
        m.price = self._preproc_prices(m.price, p_ref, p_lower_trunc=-999, p_upper_trunc=999)
        m = m.iloc[1:]
        m.price = m.price.astype(int)

        # # DIRECTION
        # m.direction = ((m.direction + 1) / 2).astype(int)

        # change column order
        # m = m[['order_id', 'event_type', 'direction', 'price_abs', 'price', 'size',
        #        'delta_t_s', 'delta_t_ns', 'time_s', 'time_ns']]
        m = m[['id', 'type', 'side', 'price_abs', 'price', 'size',
               'delta_t_s', 'delta_t_ns', 'time_s', 'time_ns',
               'cancSize', 'execSize', 'oldId', 'oldSize', 'oldPrice']]

        # add time elements of original message as feature and process NaNs
        # for all referential order types ('E','C','D','R')
        m = self._add_orig_msg_features(
            m,
            # modif_fields=['price', 'size', 'time_s', 'time_ns'])
            modif_fields=['time_s', 'time_ns'])

        assert len(m) + 1 == len(b), "length of messages (-1) and book states don't align"

        # TODO: prepend column with ticker ID
        return m.values

    def _preproc_prices(self, p, p_ref, p_lower_trunc=-10, p_upper_trunc=13):
        """ Takes prices series and reference price (best bid or mid price), 
            encoding prices relative to reference price.
            Returns scaled price series
        """
        # encode prices relative to (previous) reference price
        p = p - p_ref
        # truncate price at deviation of x
        # min tick is 100, hence min 10-level diff is 900
        # <= 1000 covers ~99.54% on bid side, ~99.1% on ask size (GOOG)
        pct_changed = 100 * len(p.loc[p > p_upper_trunc]) / len(p)
        print(f"truncating {pct_changed:.4f}% of prices > {p_upper_trunc}")
        p.loc[p > p_upper_trunc] = p_upper_trunc
        pct_changed = 100 * len(p.loc[p < p_lower_trunc]) / len(p)
        print(f"truncating {pct_changed:.4f}% of prices < {p_lower_trunc}")
        p.loc[p < p_lower_trunc] = p_lower_trunc
        # scale prices to min ticks size differences
        p /= 1
        return p

    def _add_orig_msg_features(
            self,
            m,
            modif_types={'E','C','D'},
            modif_types_special={'R'},
            modif_fields=['time_s', 'time_ns'],
            ref_cols = ['cancSize', 'execSize', 'oldId', 'oldSize', 'oldPrice'],
            nan_val=-9999
        ):
        """ Changes representation of order cancellation (2) / deletion (3) / execution (4),
            representing them as the original message and new columns containing
            the order modification details.
            This effectively does the lookup step in past data.
            TODO: lookup missing original message data from previous days' data?
        """

        # make df that converts 'R' values to 'A' values
        r_m = m.copy()
        r_m['type'] = r_m['type'].replace('R', 'A')

        # merge changes to modif_types
        m_changes = pd.merge(
            m.loc[m.type.isin(modif_types)].reset_index(),
            r_m.loc[r_m.type == 'A', ['id'] + modif_fields],
            how='left', on='id', suffixes=['', '_ref']).set_index('index')
        
        # merge changes to modif_types_special
        m_changes_special = pd.merge(
            m.loc[m.type.isin(modif_types_special)].reset_index(),
            (r_m.loc[r_m.type == 'A', ['id'] + modif_fields]).rename(columns={'id': 'oldId'}),
            how='left', on='oldId', suffixes=['', '_ref']).set_index('index')

        # add new empty columns for referenced order
        modif_cols = [field + '_ref' for field in modif_fields]
        m[modif_cols] = nan_val

        # replace order changes by original order and additional new fields
        m.loc[m_changes.index] = m_changes
        m.loc[m_changes_special.index] = m_changes_special
        m[modif_cols] = m[modif_cols].fillna(nan_val).astype(int)

        # process other ref fields
        m[ref_cols] = m[ref_cols].fillna(nan_val).astype(int)

        return m
    
    def _numeric_str(self, num, pad=2):
        if num == 0:
            return '-00'
        elif num > 0:
            return '+' + str(num).zfill(pad)
        else:
            # minus sign counts as character
            return str(num).zfill(pad + 1)