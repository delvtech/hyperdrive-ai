from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import TYPE_CHECKING, Sequence

from agent0 import LocalChain, LocalHyperdrive
from agent0.core.base.make_key import make_private_key
from fixedpointmath import FixedPoint
from web3.types import RPCEndpoint

if TYPE_CHECKING:
    from agent0.core.hyperdrive.interactive.local_hyperdrive_agent import LocalHyperdriveAgent
    from agent0.ethpy.hyperdrive.event_types import BaseHyperdriveEvent


if __name__ == "__main__":
    ### Initialization

    # Launch a local chain object with hyperdrive deployed

    # To avoid race conditions in the db, we manually sync the db
    chain = LocalChain(config=LocalChain.Config(manual_database_sync=True))
    pool = LocalHyperdrive(config=LocalHyperdrive.Config(), chain=chain)

    agents = [
        chain.init_agent(
            base=FixedPoint(1_000_000),
            eth=FixedPoint(100),
            pool=pool,
        )
        for _ in range(5)
    ]

    # We explicitly set max approval here, as we won't be making any trades
    # with these agents on the local chain side (which automatically sets approval)
    _ = [agent.set_max_approval(pool=pool) for agent in agents]

    # After initialization, we set the chain to be manual mine mode
    chain._web3.provider.make_request(method=RPCEndpoint("evm_setAutomine"), params=[False])

    ### Make async trades
    # Need async function definition for running the background tasks
    async def run_trades(_agents: list[LocalHyperdriveAgent], _chain: LocalChain) -> Sequence[BaseHyperdriveEvent]:
        # Make trades on the client side asynchronously via threads
        # NOTE we need to do threads here because underlying functions use blocking waits
        background_tasks = asyncio.gather(
            *[asyncio.to_thread(agent.add_liquidity, base=FixedPoint(1_000)) for agent in _agents]
        )

        # Manually mine the block on the server side
        # NOTE We need to ensure the trades get submitted before we mine the block,
        # otherwise there may be deadlock (i.e., we mine first, then transactions gets submitted,
        # and background tasks gets stuck waiting for block to mine.)
        # We achieve this by looking at the number of transactions on the `pending` block,
        # and give background threads control (by calling `asyncio.sleep`) when not all expected transactions
        # are submitted.
        num_pending_txns = 0
        while num_pending_txns < len(_agents):
            num_pending_txns = _chain._web3.eth.get_block_transaction_count("pending")
            await asyncio.sleep(0.5)

        # We call `anvil_mine` to manually mine a block
        _chain._web3.provider.make_request(method=RPCEndpoint("anvil_mine"), params=[])

        # NOTE we can also use a single advance time here to mine the block instead of the above
        # NOTE we can't create checkpoints here, because the server is running in
        # manual mining mode. Ensure we don't advance time more than a checkpoint duration.
        # TODO advancing time as the block mine here runs into `out of gas` error.
        # s_chain.advance_time(time_delta=1800, create_checkpoints=False)

        # Wait for all background tasks to finish
        out = await background_tasks

        return out

    out_events = asyncio.run(run_trades(agents, chain))

    # We manually sync the db here after trades go through
    pool.sync_database()

    # View trades
    events = pool.get_trade_events()

    # All trades happen on the same block.
    print(events)
