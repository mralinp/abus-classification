import asyncio
import time


async def make_coffee():
    
    await asyncio.sleep(10)
    
    return "java"

async def make_cheese_cake():
    
    await asyncio.sleep(5)
    
    return "cheesecake"

async def main():
    start = time.time()
    breakfast_batch = asyncio.gather(make_coffee(), make_cheese_cake())
    coffee, cake = await breakfast_batch
    end = time.time()
    
    elapsed_time = end - start
    
    print(f"Your breakfast is ready after: {elapsed_time} seconds")
    
if __name__ == '__main__':
    asyncio.run(main())
