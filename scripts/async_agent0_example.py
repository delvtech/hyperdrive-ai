from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from agent0 import Chain, Hyperdrive, LocalChain, LocalHyperdrive
from agent0.core.base.make_key import make_private_key
from fixedpointmath import FixedPoint

if TYPE_CHECKING:
    from agent0.core.hyperdrive.interactive.hyperdrive_agent import HyperdriveAgent
    from agent0.ethpy.hyperdrive.event_types import BaseHyperdriveEvent


if __name__ == "__main__":
    ### Initialization

    # Launch a local chain object with hyperdrive deployed

    server_chain = LocalChain(config=LocalChain.Config(db_port=1111))
    server_pool = LocalHyperdrive(config=LocalHyperdrive.Config(), chain=server_chain)

    # Generate funded agents with private keys on the server
    # NOTE: this can also be done on the remote chain side
    agent_pks = [make_private_key() for _ in range(2)]

    server_agents = [
        server_chain.init_agent(
            private_key=pk,
            base=FixedPoint(1_000_000),
            eth=FixedPoint(100),
            pool=server_pool,
        )
        for pk in agent_pks
    ]

    # We explicitly set max approval here, as we won't be making any trades
    # with these agents on the local chain side (which automatically sets approval)
    _ = [agent.set_max_approval(pool=server_pool) for agent in server_agents]

    # Launch a client chain and pool connecting to the server
    # TODO see if we can connect to the server database and have that db be in charge of
    # syncing the db to the chain.
    client_chain = Chain(rpc_uri=server_chain.rpc_uri, config=Chain.Config(db_port=2222))
    client_pool = Hyperdrive(
        chain=client_chain, hyperdrive_address=server_pool.hyperdrive_address, config=Hyperdrive.Config()
    )

    # Initialize the client agents
    client_agents = [
        client_chain.init_agent(
            private_key=pk,
            pool=client_pool,
        )
        for pk in agent_pks
    ]

    ### Make async trades

    # Need a async function wrapper to make the trade a coroutine
    async def make_trade(agent: HyperdriveAgent) -> BaseHyperdriveEvent:
        return agent.add_liquidity(base=FixedPoint(1_000))

    # Need async function definition for running the background tasks
    async def run_trades(c_agents: list[HyperdriveAgent], s_chain: LocalChain) -> list[BaseHyperdriveEvent]:
        # Make trades on the client side asynchronously
        # Create strong references to prevent the agents from being garbage collected
        background_tasks = [asyncio.create_task(make_trade(agent)) for agent in c_agents]

        # Manually mine the block on the server side
        # NOTE We need to ensure the trades get submitted before we mine the block,
        # otherwise there may be deadlock
        # (i.e., we mine first, then transactions gets submitted,
        # and background tasks gets stuck waiting for block to mine.)
        # TODO we achieve this by sleeping for now, there's probably a better way to do this,
        # e.g., ensuring the transactions are on the pending block.
        time.sleep(1)

        # TODO we may want to expose an actual mine function,
        # but for now we advance time, which mines a block.
        # NOTE we can't create checkpoints here, because the server is running in
        # manual mining mode. Ensure we don't advance time more than a checkpoint duration.
        # s_chain.mine()
        s_chain.advance_time(time_delta=1800, create_checkpoints=False)

        # Wait for all background tasks to finish
        _ = [await task for task in background_tasks]

        # Gather results
        out = [task.result() for task in background_tasks]

        # Remove references
        background_tasks.clear()

        return out

    asyncio.run(run_trades(c_agents=client_agents, s_chain=server_chain))

    pass
