import asyncio
import time
import random

IMGCONT = 0

def get_img():
    global IMGCONT
    IMGCONT = IMGCONT + 1
    return IMGCONT

async def get_async_img():
    await asyncio.sleep(2)
    return get_img()

async def process_leds(img):
    await asyncio.sleep(2)
    return ("processed LEDS for", img)

async def main():

    img_task_B = asyncio.create_task(get_async_img())

    while True:

        img = await img_task_B
        print("taken img", img)
        img_task_A = asyncio.create_task(get_async_img())
        process_led_task = asyncio.create_task(process_leds(img))
        res = await process_led_task
        print(res)

        img = await img_task_A
        print("taken img", img)
        img_task_B = asyncio.create_task(get_async_img())
        process_led_task = asyncio.create_task(process_leds(img))
        res = await process_led_task
        print(res)

asyncio.run(main())