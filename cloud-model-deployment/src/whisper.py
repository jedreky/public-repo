from aiohttp import web
import asyncio
from tempfile import NamedTemporaryFile

import whisper

from src.utils import device, get_local_ip

LOCAL_IP = asyncio.run(get_local_ip())
routes = web.RouteTableDef()


@routes.get("/")
async def health(request):
    return web.Response(text=f"All good from {LOCAL_IP}!")


@routes.post("/transcribe/{model}")
async def transcribe(request):
    multipart = await request.multipart()

    with NamedTemporaryFile() as f:
        while True:
            part = await multipart.next()

            if part is not None:
                data = await part.read()
                f.write(data)
            else:
                break

        model = whisper.load_model(name=request.match_info["model"], device=device)
        result = model.transcribe(f.name)

    return web.json_response({"text": result["text"], "source": LOCAL_IP})


if __name__ == "__main__":
    app = web.Application()
    app.add_routes(routes)
    web.run_app(app)
