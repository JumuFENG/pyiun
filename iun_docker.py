import asyncio
from app.lofig import logger
from iun import iun


async def run():
    await iun.main()
    logger.info('iun main finished, keep process running')
    while True:
        await asyncio.sleep(60)


if __name__ == '__main__':
    asyncio.run(run())
