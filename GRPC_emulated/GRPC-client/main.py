import os
import asyncio
import uvloop
from emulatedClient import EmulatedClient

# 1) uvloop for a faster event loop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

async def run_emulated_clients(ip, port, model, total_clients, mb,
                               concurrency, hold_time):
    sem       = asyncio.Semaphore(concurrency)
    start_evt = asyncio.Event()

    async def _launch_client(cid):
        # wait until we release the barrier
        await start_evt.wait()

        # throttle so only `concurrency` run at once
        async with sem:
            client = EmulatedClient(ip, port, model, mb)
            try:
                await client.run()
            except Exception as e:
                # you can log client-specific errors here
                print(f"[Client {cid}] failed: {e}")

    # 2) spawn all tasks (they’ll all block on start_evt)
    tasks = [asyncio.create_task(_launch_client(i))
             for i in range(total_clients)]

    # 3) hold them for hold_time seconds, then release them all at once
    await asyncio.sleep(hold_time)
    print(f"🚦 Releasing {total_clients} clients simultaneously!")
    start_evt.set()

    # 4) wait for every client to finish
    await asyncio.gather(*tasks)


def main():
    ip          = os.getenv('server_ip')
    port        = os.getenv('server_port', "50051")
    model       = os.getenv('model', "cnn")
    total       = int(os.getenv('clients',      500))
    mb          = int(os.getenv('target_mb',      10))
    concurrency = int(os.getenv('concurrency',   5000))
    hold_time   = float(os.getenv('hold_time',     0.1))  # seconds

    if not ip:
        raise ValueError("server_ip must be set")

    # if os.getenv('mode', 'emulated') != 'emulated':
    #     raise ValueError("Only emulated mode is supported")

    asyncio.run(
        run_emulated_clients(ip, port, model, total, mb,
                             concurrency, hold_time)
    )


if __name__ == "__main__":
    main()
