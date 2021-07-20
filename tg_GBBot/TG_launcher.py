from typing import io
import aiogram.bot.api
import matplotlib.pyplot as plt

import config
import logging
import cv2
import numpy as np
from PIL import Image

from aiogram import Bot, Dispatcher, executor, types
from GBDetector import Detection
# bot init
bot = Bot(token=config.TOKEN)
dp = Dispatcher(bot)


# start message
@dp.message_handler(commands=['start'])
async def process_help_command(message: types.Message):
    await message.answer("Привет, я бот, определяющий породы собак по фотографии) \
                        Отправь мне фото и я скажу тебе кто на нем)")


# bot help
@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    await message.answer("Отправь мне фотографию собачки, а я назову ее породу)")


# echo bot test
@dp.message_handler()
async def echo(message: types.Message):
    await message.answer(message.text)


# photo reaction test
@dp.message_handler(content_types=['photo'])
async def photo_reaction(message):
    file_dct = await bot.get_file(file_id=message["photo"][-1]["file_id"])
    image = await bot.download_file(file_dct['file_path'])
    img = Image.open(image).convert('RGB')
    open_cv_image = np.array(img)

    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)

    Detection(open_cv_image)

    # await message.answer_photo(image.seek(0))


# run long-polling
if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
