from __future__ import annotations

import asyncio
import os
from dataclasses import asdict
from typing import TYPE_CHECKING, Sequence

from agent0 import Chain, Hyperdrive, LocalChain, LocalHyperdrive
from agent0.core.base.make_key import make_private_key
from fixedpointmath import FixedPoint
from web3.types import RPCEndpoint

if TYPE_CHECKING:
    from agent0.core.hyperdrive.interactive.hyperdrive_agent import HyperdriveAgent
    from agent0.ethpy.hyperdrive.event_types import BaseHyperdriveEvent


if __name__ == "__main__":
    ### Initialization

    # Launch a local chain object with hyperdrive deployed

    server_chain = LocalChain(config=LocalChain.Config())
    server_pool = LocalHyperdrive(config=LocalHyperdrive.Config(), chain=server_chain)

    # Generate funded agents with private keys on the server
    # NOTE: this can also be done on the remote chain side
    agent_pks = [make_private_key() for _ in range(5)]

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
    # We connect the client chain to the server chain's db
    postgres_config = asdict(server_chain.postgres_config)
    # TODO `use_existing_postgres` reads from the environment (and an `.env` file), so we set it here.
    # Ideally we can pass it in as a config
    for k, v in postgres_config.items():
        os.environ[k] = str(v)

    client_chain = Chain(
        rpc_uri=server_chain.rpc_uri,
        config=Chain.Config(
            use_existing_postgres=True,
        ),
    )
    client_pool = Hyperdrive(
        chain=client_chain,
        hyperdrive_address=server_pool.hyperdrive_address,
        config=Hyperdrive.Config(),
    )

    # Initialize the client agents
    client_agents = [
        client_chain.init_agent(
            private_key=pk,
            pool=client_pool,
        )
        for pk in agent_pks
    ]

    ## After initialization, we set the chain to be manual mine mode
    server_chain._web3.provider.make_request(method=RPCEndpoint("evm_setAutomine"), params=[False])

    ### Make async trades

    # Need async function definition for running the background tasks
    async def run_trades(c_agents: list[HyperdriveAgent], s_chain: LocalChain) -> Sequence[BaseHyperdriveEvent]:
        # Make trades on the client side asynchronously via threads
        # NOTE we need to do threads here because underlying functions use blocking waits
        background_tasks = asyncio.gather(
            *[asyncio.to_thread(agent.add_liquidity, base=FixedPoint(1_000)) for agent in c_agents]
        )

        # Manually mine the block on the server side
        # NOTE We need to ensure the trades get submitted before we mine the block,
        # otherwise there may be deadlock
        # (i.e., we mine first, then transactions gets submitted,
        # and background tasks gets stuck waiting for block to mine.)
        # TODO we achieve this by sleeping for now, there's probably a better way to do this,
        # e.g., ensuring the transactions are on the pending block.
        # NOTE this sleep must be asyncio sleep to give control to threads to submit the trades.
        # `time.sleep` doesn't work here. In general, there's some strict order of operations here
        # that gets complicated, likely due to the GIL.
        await asyncio.sleep(1)

        # TODO we may want to expose an actual mine function,
        # but for now we advance time, which mines a block.
        # NOTE we can't create checkpoints here, because the server is running in
        # manual mining mode. Ensure we don't advance time more than a checkpoint duration.
        s_chain._web3.provider.make_request(method=RPCEndpoint("anvil_mine"), params=[])
        # s_chain.advance_time(time_delta=1800, create_checkpoints=False)

        # Wait for all background tasks to finish
        out = await background_tasks

        return out

    out_events = asyncio.run(run_trades(c_agents=client_agents, s_chain=server_chain))

    # View trades
    # NOTE since the trades happened on the client object, and remote objects do lazy db updating,
    # we need to either explicitly sync the server pool events, or reference events from the
    # client pool (which automatically syncs events when called).
    server_pool._run_blocking_data_pipeline()
    events = server_pool.get_trade_events()
    pass
