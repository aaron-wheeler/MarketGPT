# retroactively collect each simulated book state from the simulated messages

from __future__ import annotations
from typing import Tuple, List, Optional
import os
import sys
import argparse
import time
import csv
import numpy as np
from glob import glob

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'simulator'))
sys.path.append(os.path.join(ROOT_DIR, 'equities/data_processing'))

from simulator.core import Message
from simulator.markets.order_book import OrderBook
from simulator.markets.orders import LimitOrder, Side, MarketOrder
from equities.data_processing import itch_preproc


Level = Tuple[int, int]   # (price, volume)


def make_columns(price_levels: int) -> List[str]:
    cols = ["time"]
    for lvl in range(1, price_levels + 1):
        cols += [f"{lvl}_bid_price", f"{lvl}_bid_vol", f"{lvl}_ask_price", f"{lvl}_ask_vol"]
    return cols


def pad_only(levels: List[Level], price_levels: int) -> List[Optional[Level]]:
    """
    Assumes levels <= price_levels.
    Pads with (None, None) to exactly price_levels.
    """
    if len(levels) > price_levels:
        raise ValueError(f"Input has {len(levels)} levels > price_levels={price_levels}.")
    return levels + [(None, None)] * (price_levels - len(levels))


def snapshot_to_row(time_val: int,
                    bids: List[Level],
                    asks: List[Level],
                    price_levels: int) -> List[Optional[int]]:
    """
    Build one row: [time, 1_bid_price, 1_bid_vol, 1_ask_price, 1_ask_vol, ..., N_*].
    Outputs Python ints or None (so CSV writes blanks for missing).
    """
    bids_p = pad_only(bids, price_levels)
    asks_p = pad_only(asks, price_levels)

    row: List[Optional[int]] = [int(time_val)]
    for (bp, bv), (ap, av) in zip(bids_p, asks_p):
        row.extend([bp, bv, ap, av])
    return row


def process_order_message(msg: np.ndarray,
                          order_book: OrderBook,
                          prev_time: int,
                          tickers: dict[int, str],
                          agent_id: int = 1):
    symbol = tickers[msg[0]]
    order_id = msg[1]
    event_type = msg[2]
    price = msg[4]
    time = prev_time + (msg[8]*1000000000) + msg[9]

    # handle order based on event type
    if event_type == 1:
        # ADD LIMIT ORDER
        direction = Side.BID if msg[3] == 0 else Side.ASK
        fill_size = msg[6]
        order = LimitOrder(
            order_id=order_id,
            agent_id=agent_id,
            time_placed=time,
            symbol=symbol,
            quantity=fill_size,
            side=direction,
            limit_price=price,
        )
        order_book.handle_limit_order(order)
    elif event_type == 2:
        # EXECUTE ORDER
        fill_size = msg[6]
        direction = Side.BID if msg[3] == 1 else Side.ASK # opposite of direction in non-execution messages
        order = MarketOrder(
            order_id=order_id,
            agent_id=agent_id,
            time_placed=time,
            symbol=symbol,
            quantity=fill_size,
            side=direction, # Buy Order if Side.BID (remove liquidity from ask side), Sell Order if Side.ASK (remove liquidity from bid side)
        )
        order_book.handle_market_order(order)
    elif event_type == 3:
        # EXECUTE ORDER WITH PRICE DIFFERENT THAN DISPLAY
        # This order type is most likely an execution of a price-to-comply order, which is handled by the simulator
        # but this not encoded in the ITCH data beforehand, so we cannot know whether an order is price-to-comply at the time of submission
        # therefore, we handle this event type as a modifed order and then regular execution order (for now, until we revise the data processing)

        # modfify the matched limit order
        direction = Side.BID if msg[3] == 0 else Side.ASK
        ref_order_time = (msg[15] * 1000000000) + msg[16]
        ref_order_size = msg[14]
        ref_order_price = msg[17]
        # define original order
        original_order = LimitOrder(
            order_id=order_id,
            agent_id=agent_id,
            time_placed=ref_order_time,
            symbol=symbol,
            quantity=ref_order_size,
            side=direction,
            limit_price=ref_order_price,
        )
        # define modified order
        modified_order = LimitOrder(
            order_id=order_id,
            agent_id=agent_id,
            time_placed=ref_order_time,
            symbol=symbol,
            quantity=ref_order_size,
            side=direction,
            limit_price=price,
        )
        order_book.modify_order(original_order, modified_order)
        # execute the modified order
        fill_size = msg[6]
        direction = Side.BID if msg[3] == 1 else Side.ASK # opposite of direction in non-execution messages
        order = MarketOrder(
            order_id=order_id,
            agent_id=agent_id,
            time_placed=time,
            symbol=symbol,
            quantity=fill_size,
            side=direction, # Buy Order if Side.BID (remove liquidity from ask side), Sell Order if Side.ASK (remove liquidity from bid side)
        )
        order_book.handle_market_order(order)
    elif event_type == 4:
        # CANCEL ORDER
        direction = Side.BID if msg[3] == 0 else Side.ASK
        ref_order_time = (msg[15] * 1000000000) + msg[16]
        if msg[7] == 0:
            # FULL DELETION
            fill_size = msg[6]
            order = LimitOrder(
                order_id=order_id,
                agent_id=agent_id,
                time_placed=ref_order_time,
                symbol=symbol,
                quantity=fill_size,
                side=direction,
                limit_price=price,
            )
            order_book.cancel_order(order)
        else:
            # PARTIAL CANCELLATION
            cancel_size = msg[6]
            ref_order_size = msg[7] + cancel_size # total size of order before partial cancel
            order = LimitOrder(
                order_id=order_id,
                agent_id=agent_id,
                time_placed=ref_order_time,
                symbol=symbol,
                quantity=ref_order_size,
                side=direction,
                limit_price=price,
            )
            order_book.partial_cancel_order(order, cancel_size)
    elif event_type == 5:
        # REPLACE ORDER
        direction = Side.BID if msg[3] == 0 else Side.ASK
        old_order_id = msg[12]
        old_order_time = (msg[15] * 1000000000) + msg[16]
        old_order_size = msg[14]
        old_order_price = msg[17] # old_price_abs (not mid_price so we cannot calculate using price_ref msg[13])
        # define old order
        old_order = LimitOrder(
            order_id=old_order_id,
            agent_id=agent_id,
            time_placed=old_order_time,
            symbol=symbol,
            quantity=old_order_size,
            side=direction,
            limit_price=old_order_price,
        )
        new_order_size = msg[6]
        # define new order
        new_order = LimitOrder(
            order_id=order_id,
            agent_id=agent_id,
            time_placed=time,
            symbol=symbol,
            quantity=new_order_size,
            side=direction,
            limit_price=price,
        )
        order_book.replace_order(agent_id, old_order, new_order)
    else:
        raise NotImplementedError("Event type not implemented")
    
    return


def collect_simulated_book_data(sim_data_dir: str,
                                itch_msg_file: str,
                                proc_msg_file: str,
                                symbols_file: str,
                                symbol: str,
                                price_levels: int = 20):
    # collect metadata
    meta_data_file = glob(sim_data_dir + '/*metadata*.txt')[0]
    with open(meta_data_file, "r") as f:
        for line in f:
            if line.startswith("msgs_to_load_LOB:"):
                msgs_to_load_LOB = int(line.split(":")[1].strip())
            elif line.startswith("num_context_msgs:"):
                num_context_msgs = int(line.split(":")[1].strip())

    # init ABIDES exchange agent
    TIME = 0
    WORLD_AGENT_ID = 1

    class FakeExchangeAgent:
        def __init__(self):
            self.messages = []
            self.current_time = TIME
            self.mkt_open = TIME
            self.book_logging = None
            self.stream_history = 10

        def reset(self):
            self.messages = []

        def send_message(self, recipient_id: int, message: Message, _: int = 0):
            self.messages.append((recipient_id, message))

        def logEvent(self, *args, **kwargs):
            pass

    # create reverse ticker symbol mapping (key is index, value is ticker)
    tickers = {}
    with open(symbols_file) as f:
        idx = 0
        for line in f:
            idx += 1
            tickers[idx] = line.strip()
    
    # load data needed to init LOB
    first_message = (itch_preproc.load_message_df(itch_msg_file)).iloc[0]
    proc_messages = np.array(np.load(proc_msg_file, mmap_mode='r')[0:(msgs_to_load_LOB + num_context_msgs)])

    # init new book under nasdaq agent
    nasdaq_agent = FakeExchangeAgent()
    order_book = OrderBook(nasdaq_agent, symbol)

    # insert first bid order into LOB
    bid_order = LimitOrder(
        order_id=first_message['id'],
        agent_id=WORLD_AGENT_ID,
        time_placed=first_message['time'],
        symbol=symbol,
        quantity=int(first_message['size']),
        side=Side.BID if first_message['side'] == 0 else Side.ASK,
        limit_price=int(first_message['price']*100),
    )
    order_book.handle_limit_order(bid_order)

    # init variables to keep track of previous time, price, etc.
    prev_time = first_message['time']

    # iterate through context messages and set up pre-generation order book
    for msg in proc_messages:
        # process order message
        process_order_message(msg, order_book, prev_time, tickers)
        prev_time += (msg[8]*1000000000) + msg[9]
    print("Processed all context messages")

    # load simulated messages
    sim_messages = np.loadtxt(glob(sim_data_dir + '/*gen_data*.csv')[0], delimiter=',', skiprows=1, dtype=int)

    # prepare CSV file for writing
    columns = make_columns(price_levels)
    f = open(sim_data_dir + "/gen_book_data_full_L2.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(columns)

    # record starting LOB state
    L2_bid_data = order_book.get_l2_bid_data(price_levels)
    L2_ask_data = order_book.get_l2_ask_data(price_levels)
    row = snapshot_to_row(prev_time, L2_bid_data, L2_ask_data, price_levels)
    writer.writerow(row)

    # iterate through generated messages and collect order book snapshots
    for msg in sim_messages:
        process_order_message(msg, order_book, prev_time, tickers)
        prev_time += (msg[8]*1000000000) + msg[9]
        L2_bid_data = order_book.get_l2_bid_data(price_levels)
        L2_ask_data = order_book.get_l2_ask_data(price_levels)
        row = snapshot_to_row(prev_time, L2_bid_data, L2_ask_data, price_levels)
        writer.writerow(row)
    f.close()
    print("Processed all simulated messages")


if __name__ == "__main__":
    itch_msg_path = os.path.join(ROOT_DIR, 'dataset/raw/ITCH/12302019.NASDAQ_ITCH50_AAPL_message.csv')
    proc_msg_path = os.path.join(ROOT_DIR, 'dataset/proc/ITCH/multi/pre_train/full_view/12302019.NASDAQ_ITCH50_AAPL_message_proc.npy')
    symbols_file_path = os.path.join(ROOT_DIR, 'dataset/symbols/custom_symbols.txt')

    parser = argparse.ArgumentParser(description="Collect simulated book data")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing simulation data")
    parser.add_argument("--itch_msg_file", default=itch_msg_path, type=str, help="Path to the unprocessed message file")
    parser.add_argument("--proc_msg_file", default=proc_msg_path, type=str, help="Path to the processed message file")
    parser.add_argument("--symbols_file", default=symbols_file_path, type=str, help="Path to the symbols directory file")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Symbol for the stock")
    parser.add_argument("--price-levels", type=int, default=20, help="Number of price levels to collect (per side)")
    args = parser.parse_args()

    start_time = time.time()
    collect_simulated_book_data(
        sim_data_dir=args.dir,
        itch_msg_file=args.itch_msg_file,
        proc_msg_file=args.proc_msg_file,
        symbols_file=args.symbols_file,
        symbol=args.symbol,
        price_levels=args.price_levels
    )
    end_time = time.time()
    print(f"Time taken to collect simulated book data: {end_time - start_time} seconds")