import aiohttp
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


async def get_local_ip():
    host = "http://169.254.169.254/latest"
    async with aiohttp.ClientSession() as session:
        async with session.put(
            f"{host}/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        ) as resp:
            print(resp.status)
            token = await resp.text()

        async with session.get(
            f"{host}/meta-data/local-ipv4", headers={"X-aws-ec2-metadata-token": token}
        ) as resp:
            print(resp.status)
            local_ip = await resp.text()

    return local_ip
