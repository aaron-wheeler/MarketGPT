from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from decimal import Decimal

from equities.data_processing.itch_encoding import Vocab, Message_Tokenizer

import os
import sys

# def transform_L2_state(
#         book: np.array,
#         price_levels: int,
#         tick_size: int = 100,
#         #divide_by: int = 1,
#     ) -> np.array:
#     """ Transformation for data loading:
#         Converts L2 book state from data to price_levels many volume
#         series used as input to the model. The first element (column) of the
#         input and output is the change in mid price.
#         Converts sizes to negative sizes for ask side (sell orders).
#     """
#     delta_p_mid, book = book[:1], book[1:]
#     book = book.reshape((-1,2))
#     mid_price = np.ceil((book[0, 0] + book[1, 0]) / (2*tick_size)).__mul__(tick_size).astype(int)
#     book = book.at[:, 0].set((book[:, 0] - mid_price) // tick_size)
#     # change relative prices to indices
#     book = book.at[:, 0].set(book[:, 0] + price_levels // 2)
#     # set to out of bounds index, so that we can use -1 to indicate nan
#     # out of bounds will be ignored in setting value in jax
#     book = np.where(book < 0, -price_levels-1, book)

#     mybook = np.zeros(price_levels, dtype=np.int32)
#     mybook = mybook.at[book[:, 0]].set(book[:, 1])
    
#     # set ask volume to negative (sell orders)
#     mybook = mybook.at[price_levels // 2:].set(mybook[price_levels // 2:] * -1)
#     mybook = np.concatenate((
#         delta_p_mid.astype(np.float32),
#         mybook.astype(np.float32) / 1000
#     ))

#     # return mybook.astype(np.float32) #/ divide_by
#     return mybook 


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
        save_dir: str,
        filter_above_lvl: Optional[int] = None,
        skip_existing: bool = False,
        remove_premarket: bool = True,
        remove_aftermarket: bool = True,
    ) -> None:

    v = Vocab()
    tok = Message_Tokenizer()

    assert len(message_files) == len(book_files)
    for m_f, b_f in tqdm(zip(message_files, book_files)):
        print(m_f)
        m_path = save_dir + m_f.rsplit('/', maxsplit=1)[-1][:-4] + '_proc.npy'
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
        save_dir: str,
        n_price_series: int,
        filter_above_lvl: Optional[int] = None,
        allowed_events=['A','E','C','D','R'],
        skip_existing: bool = False,
        use_raw_book_repr=False,
        remove_premarket: bool = True,
        remove_aftermarket: bool = True,
    ) -> None:

    for m_f, b_f in tqdm(zip(message_files, book_files)):
        print(m_f)
        print(b_f)
        b_path = save_dir + b_f.rsplit('/', maxsplit=1)[-1][:-4] + '_proc.npy'
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

        np.save(b_path, book, allow_pickle=True)

def process_book(
        b: pd.DataFrame,
        price_levels: int
    ) -> np.ndarray:

    # mid-price rounded to nearest tick
    p_ref = ((b.iloc[:, 0] + b.iloc[:, 2]) / 2).mul(100).round().astype(int)
    b_indices = b.iloc[:, ::2].mul(100).sub(p_ref, axis=0).astype(int)
    b_indices = b_indices + price_levels // 2 # make tick differences fit between span of 0 to price_levels
    b_indices.columns = list(range(b_indices.shape[1])) # reset col indices
    vol_book = b.iloc[:, 1::2].copy().astype(int)
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
    # TODO: prepend column with ticker ID
    return np.concatenate([mid_diff[:, None], mybook], axis=1)

if __name__ == '__main__':
    parent_folder_path, current_dir = os.path.split(os.path.abspath(''))
    load_path = parent_folder_path + '/' + current_dir + '/dataset/raw/ITCH/'
    save_path = parent_folder_path + '/' + current_dir + '/dataset/ITCH/'

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
    args = parser.parse_args()

    assert not (args.messages_only and args.book_only)

    message_files = sorted(glob(args.data_dir + '*message*.csv'))
    book_files = sorted(glob(args.data_dir + '*book*.csv'))

    print('found', len(message_files), 'message files')
    print('found', len(book_files), 'book files')
    print()

    if not args.book_only:
        print('processing messages...')
        process_message_files(
            message_files,
            book_files,
            args.save_dir,
            filter_above_lvl=args.filter_above_lvl,
            skip_existing=args.skip_existing,
        )
    else:
        print('Skipping message processing...')
    print()
    
    if not args.messages_only:
        print('processing books...')
        process_book_files(
            message_files,
            book_files,
            args.save_dir,
            filter_above_lvl=args.filter_above_lvl,
            n_price_series=args.n_tick_range,
            skip_existing=args.skip_existing,
            use_raw_book_repr=args.use_raw_book_repr,
        )
    else:
        print('Skipping book processing...')
    print('DONE')