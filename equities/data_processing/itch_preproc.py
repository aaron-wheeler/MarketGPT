"""
This code was built on the work of the authors of the paper "Generative AI for End-to-End Limit Order Book Modelling: A Token-Level Autoregressive Generative Model of Message Flow Using a Deep State Space Network"

The original code can be found at: https://github.com/peernagy/LOBS5/blob/main/lob/preproc.py

I refactored the code to work with the ITCH datasets (with my custom fields) and the 'itch_encoding.py' file.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from decimal import Decimal

from itch_encoding import Vocab, Message_Tokenizer

import os
import sys


def load_message_df(m_f: str) -> pd.DataFrame:
    # cols = ['time', 'event_type', 'order_id', 'size', 'price', 'direction']
    # cols = ['time','type','id','side','size','price','cancSize','execSize','oldId','oldSize','oldPrice','mpid']
    messages = pd.read_csv(
        m_f,
        # names=cols,
        # usecols=cols,
        # index_col=False,
        # dtype={
        #     #'time': 'float64',
        #     'time': 'int32',
        #     'type': str,
        #     'id': 'int32',
        #     'side': 'int32',
        #     'size': 'int32',
        #     'price': 'int32',
        #     'cancSize': 'int32', # may be NaN
        #     'execSize': 'int32', # may be NaN
        #     'oldId': 'int32', # may be NaN
        #     'oldSize': 'int32', # may be NaN
        #     'oldPrice': 'int32', # may be NaN
        #     'mpid': str # may be NaN
        # }
    )
    # messages.time = messages.time.apply(lambda x: Decimal(x))
    return messages


def process_message_files(
        message_files: list[str],
        book_files: list[str],
        symbols_file: str,
        save_dir: str,
        filter_above_lvl: Optional[int] = None,
        skip_existing: bool = False,
        remove_premarket: bool = False,
        remove_aftermarket: bool = False,
    ) -> None:

    v = Vocab()
    tok = Message_Tokenizer()

    # create ticker symbol mapping
    tickers = {}
    with open(symbols_file) as f:
        idx = 0
        for line in f:
            idx += 1
            tickers[line.strip()] = idx

    assert len(message_files) == len(book_files)
    for m_f, b_f in tqdm(zip(message_files, book_files)):
        print(m_f)
        m_path = save_dir + m_f.rsplit('/', maxsplit=1)[-1][:-4] + '_proc.npy'
        symbol = m_f.rsplit('/', maxsplit=1)[-1][:-12].rsplit('_', maxsplit=1)[-1]
        if skip_existing and Path(m_path).exists():
            print('skipping', m_path)
            continue
        
        messages = load_message_df(m_f)

        book = pd.read_csv(
            b_f,
            # index_col=False,
            # header=None
        )
        assert len(messages) == len(book)

        if filter_above_lvl:
            book = book.iloc[:, :filter_above_lvl * 4 + 1]
            messages, book = filter_by_lvl(messages, book, filter_above_lvl)

        # remove mpid field from ITCH data
        messages = messages.drop(columns=['mpid'])

        # remove pre-market and after-market hours from ITCH data
        if remove_premarket:
            messages = messages[messages['time'] >= 34200000000000]
        if remove_aftermarket:
            messages = messages[messages['time'] <= 57600000000000]

        # format time for pre-processing
        messages['time'] = messages['time'].astype('string')
        messages['time'] = messages['time'].apply(lambda x: '.'.join((x[0:5], x[5:])))
        messages['time'] = messages['time'].apply(lambda x: Decimal(x))

        # convert price to pennies from dollars
        messages['price'] = (messages['price'] * 100).astype('int')
        messages['oldPrice'] = (messages['oldPrice'] * 100) # make int after dealing with NaNs

        # # convert replace 'R' events to cancel 'D' and add 'A' events
        # rows_list = []
        # for index, row in messages.iterrows():
        #     if row['type'] == 'R':
        #         # create cancel event..
        #         order_elements = messages.loc[index]
        #         cancel_dict = {'time': order_elements.time, 'type': 'D', 'id': (order_elements.oldId).astype('int'), 'side': order_elements.side, 'size': 0.0, 'price': (order_elements.oldPrice).astype('int'), 'cancSize': order_elements.oldSize, 'execSize': order_elements.execSize, 'oldId': order_elements.execSize, 'oldSize': order_elements.execSize, 'oldPrice': order_elements.execSize}
        #         # ..add it to the list
        #         rows_list.append(cancel_dict)
                
        #         # create add event..
        #         add_dict = {'time': order_elements.time, 'type': 'A', 'id': (order_elements.id).astype('int'), 'side': order_elements.side, 'size': order_elements.size, 'price': (order_elements.price).astype('int'), 'cancSize': order_elements.execSize, 'execSize': order_elements.execSize, 'oldId': order_elements.execSize, 'oldSize': order_elements.execSize, 'oldPrice': order_elements.execSize}
        #         # ..add it to the list
        #         rows_list.append(add_dict)
        #     else:
        #         # add the original event to the list
        #         rows_list.append(messages.loc[index].to_dict())
        # # create a new dataframe from the list
        # messages = pd.DataFrame(rows_list) # TODO: book files no longer match up with messages... need to fix this

        
        print('<< pre processing >>')
        m_ = tok.preproc(messages, book)

        # prepend column with ticker ID
        ticker_id = tickers[symbol]
        m_ = np.concatenate([np.full((m_.shape[0], 1), ticker_id), m_], axis=1)

        # save processed messages
        np.save(m_path, m_)
        print('saved to', m_path)

def get_price_range_for_level(
        book: pd.DataFrame,
        lvl: int
    ) -> pd.DataFrame:
    assert lvl > 0
    assert lvl <= (book.shape[1] // 4)
    p_range = book.iloc[:, [(lvl-1) * 4 + 1, (lvl-1) * 4 + 3]] # lvl bid and ask prices
    p_range.columns = ['p_min', 'p_max']
    return p_range

def filter_by_lvl(
        messages: pd.DataFrame,
        book: pd.DataFrame,
        lvl: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

    assert messages.shape[0] == book.shape[0]
    p_range = get_price_range_for_level(book, lvl)
    messages = messages[(messages.price <= p_range.p_max) & (messages.price >= p_range.p_min)]
    book = book.loc[messages.index]
    return messages, book


def process_book_files(
        message_files: list[str],
        book_files: list[str],
        symbols_file: str,
        save_dir: str,
        n_price_series: int,
        filter_above_lvl: Optional[int] = None,
        allowed_events=['A','E','C','D','R'],
        skip_existing: bool = False,
        use_raw_book_repr=False,
        remove_premarket: bool = False,
        remove_aftermarket: bool = False,
    ) -> None:

    # create ticker symbol mapping
    tickers = {}
    with open(symbols_file) as f:
        idx = 0
        for line in f:
            idx += 1
            tickers[line.strip()] = idx

    # process and save each book file
    for m_f, b_f in tqdm(zip(message_files, book_files)):
        print(m_f)
        print(b_f)
        b_path = save_dir + b_f.rsplit('/', maxsplit=1)[-1][:-4] + '_proc.npy'
        symbol = m_f.rsplit('/', maxsplit=1)[-1][:-12].rsplit('_', maxsplit=1)[-1]
        if skip_existing and Path(b_path).exists():
            print('skipping', b_path)
            continue

        messages = load_message_df(m_f)

        book = pd.read_csv(
            b_f,
            # index_col=False,
            # header=None
        )

        # remove pre-market and after-market hours from ITCH data
        if remove_premarket:
            messages = messages[messages['time'] >= 34200000000000]
        if remove_aftermarket:
            messages = messages[messages['time'] <= 57600000000000]

        # remove disallowed order types
        messages = messages.loc[messages.type.isin(allowed_events)]
        # make sure book is same length as messages
        book = book.loc[messages.index]

        if filter_above_lvl is not None:
            messages, book = filter_by_lvl(messages, book, filter_above_lvl)

        # remove time field from ITCH book data
        book = book.drop(columns=['time'])

        assert len(messages) == len(book)

        # convert to n_price_series separate volume time series (each tick is a price level)
        if not use_raw_book_repr:
            book = process_book(book, price_levels=n_price_series)
        else:
            # prepend delta mid price column to book data
            p_ref = ((book.iloc[:, 0] + book.iloc[:, 2]) / 2).mul(100).round().astype(int)
            mid_diff = p_ref.diff().fillna(0).astype(int)
            book = np.concatenate((mid_diff.values.reshape(-1,1), book.values), axis=1)

        # prepend column with ticker ID
        ticker_id = tickers[symbol]
        book = np.concatenate([np.full((book.shape[0], 1), ticker_id), book], axis=1)

        # save processed book
        np.save(b_path, book, allow_pickle=True)
        print('saved to', b_path)

def process_book(
        b: pd.DataFrame,
        price_levels: int
    ) -> np.ndarray:

    # mid-price rounded to nearest tick
    p_ref = ((b.iloc[:, 0] + b.iloc[:, 2]) / 2)
    # determine if NaN is present
    if p_ref.isnull().values.any():
        print('NaN detected. Replacing with best existing bid or ask price.')
        # replace mid-price with best existing bid or ask price
        p_ref = p_ref.fillna(b.iloc[:, 0].combine_first(b.iloc[:, 2]))
        # replace any remaining NaNs with previous value
        if p_ref.isnull().values.any():
            print('More NaN detected. Replacing with previous value.')
            p_ref = p_ref.ffill()
    p_ref = p_ref.mul(100).round().astype(int) # format; round to nearest tick
    # how far are bid and ask from mid price?
    b_indices = b.iloc[:, ::2].mul(100).fillna(0).sub(p_ref, axis=0).astype(int)
    b_indices = b_indices + price_levels // 2 # valid tick differences will fit between span of 0 to price_levels
    b_indices.columns = list(range(b_indices.shape[1])) # reset col indices
    vol_book = b.iloc[:, 1::2].copy().fillna(0).astype(int)
    # convert sell volumes (ask side) to negative
    vol_book.iloc[:, 1::2] = vol_book.iloc[:, 1::2].mul(-1)
    vol_book.columns = list(range(vol_book.shape[1])) # reset col indices

    # convert to book representation with volume at each price level relative to reference price (mid)
    # whilst preserving empty levels to maintain sparse representation of book
    # i.e. at each time we have a fixed width snapshot around the mid price
    # therefore movement of the mid price needs to be a separate feature (e.g. relative to previous price)

    mybook = np.zeros((len(b), price_levels), dtype=np.int32)

    a = b_indices.values
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            price = a[i, j]
            # remove prices outside of price_levels range
            if price >= 0 and price < price_levels:
                mybook[i, price] = vol_book.values[i, j]

    # prepend column with best bid changes (in ticks)
    mid_diff = p_ref.diff().fillna(0).astype(int).values
    return np.concatenate([mid_diff[:, None], mybook], axis=1)

if __name__ == '__main__':
    parent_folder_path, current_dir = os.path.split(os.path.abspath(''))
    load_path = parent_folder_path + '/' + current_dir + '/dataset/raw/ITCH/'
    save_path = parent_folder_path + '/' + current_dir + '/dataset/proc/ITCH/'
    symbols_load_path = parent_folder_path + '/' + current_dir + '/dataset/symbols/'

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=load_path,
		     			help="where to load data from")
    parser.add_argument("--save_dir", type=str, default=save_path,
		     			help="where to save processed data")
    parser.add_argument("--filter_above_lvl", type=int,
                        help="filters down from levels present in the data to specified number of price levels")
    parser.add_argument("--n_tick_range", type=int, default=500,
                        help="how many ticks price series should be calculated")
    parser.add_argument("--skip_existing", action='store_true', default=False)
    parser.add_argument("--messages_only", action='store_true', default=False)
    parser.add_argument("--book_only", action='store_true', default=False)
    parser.add_argument("--use_raw_book_repr", action='store_true', default=False)
    parser.add_argument("--remove_premarket", action='store_true', default=False)
    parser.add_argument("--remove_aftermarket", action='store_true', default=False)
    args = parser.parse_args()

    assert not (args.messages_only and args.book_only)

    message_files = sorted(glob(args.data_dir + '*message*.csv'))
    book_files = sorted(glob(args.data_dir + '*book*.csv'))
    symbols_file = sorted(glob(symbols_load_path + '*sp500*.txt'))[0]

    print('found', len(message_files), 'message files')
    print('found', len(book_files), 'book files')
    print()

    if not args.book_only:
        print('processing messages...')
        process_message_files(
            message_files,
            book_files,
            symbols_file,
            args.save_dir,
            filter_above_lvl=args.filter_above_lvl,
            skip_existing=args.skip_existing,
            remove_premarket=args.remove_premarket,
            remove_aftermarket=args.remove_aftermarket,
        )
    else:
        print('Skipping message processing...')
    print()
    
    if not args.messages_only:
        print('processing books...')
        process_book_files(
            message_files,
            book_files,
            symbols_file,
            args.save_dir,
            filter_above_lvl=args.filter_above_lvl,
            n_price_series=args.n_tick_range,
            skip_existing=args.skip_existing,
            use_raw_book_repr=args.use_raw_book_repr,
            remove_premarket=args.remove_premarket,
            remove_aftermarket=args.remove_aftermarket,
        )
    else:
        print('Skipping book processing...')
    print('DONE')