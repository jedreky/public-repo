import aiohttp
import asyncio
import os
import time

ENDPOINT = os.getenv("ENDPOINT", default="localhost")


async def single_call(session, model, files, idx):
    print(f"Submitted request: {idx}")
    async with session.post(
        f"http://{ENDPOINT}:8080/transcribe/{model}", data=files
    ) as resp:
        print(f"Received response from {idx}, status: {resp.status}")
        print(await resp.text())


async def main(model):
    if model == "base":
        reps = [5, 20]
    else:
        reps = [3, 10]

    print(f"Testing model: {model}")
    with open("speech_sample.mp3", "rb") as f:
        data = f.read()

    files = {"file": data}

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=20 * 60)
    ) as session:
        print("Warm-up phase")
        tasks = [
            asyncio.ensure_future(single_call(session, model, files, idx))
            for idx in range(reps[0])
        ]
        await asyncio.gather(*tasks)

        print("Main run")
        t0 = time.time()
        tasks = [
            asyncio.ensure_future(single_call(session, model, files, idx))
            for idx in range(reps[1])
        ]
        await asyncio.gather(*tasks)
        t_elapsed = time.time() - t0
        print(f"Summary for model {model}")
        print(f"Total time elapsed: {int(t_elapsed)} secs")
        t_ratio = (reps[1] * 12) / t_elapsed
        print(f"Audio length to processing time ratio: {t_ratio:.1f}")


if __name__ == "__main__":
    for model in ["base", "medium"]:
        asyncio.run(main(model))
