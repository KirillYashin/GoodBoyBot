from typing import io
from io import BytesIO
import aiogram.bot.api
import matplotlib.pyplot as plt

import config
import logging
import cv2
import numpy as np
from PIL import Image

from aiogram import Bot, Dispatcher, executor, types
from GBDetector import Detector, get_breed_info, translator
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


# photo reaction test
@dp.message_handler(content_types=['photo'])
async def photo_reaction(message):
    file_dct = await bot.get_file(file_id=message["photo"][-1]["file_id"])
    image = await bot.download_file(file_dct['file_path'])
    img = Image.open(image).convert('RGB')
    open_cv_image = np.array(img)

    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    Dogs = []
    Cats = []
    Conf = []
    Dogs, Cats, Conf, out = Detector(open_cv_image)
    out = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    if len(Dogs) == 0:
        if len(Cats) == 0:
            await message.answer('Я никого не нашел на фото(')
        else:
            await message.answer('Кис кис, кис кис. Я ботик, ты котик)')
    elif len(Dogs) == 1:
        output_image = Image.fromarray(out)
        stream_image = BytesIO()
        stream_image.name = "Dogs.jpg"
        output_image.save(stream_image, 'JPEG')
        stream_image.seek(0)
        breed = translator(Dogs[0][0])
        breed_info, image_url = get_breed_info(breed)
        breed_info = "Мне кажется, что это ", breed + '\n' + breed_info
        await message.answer_photo(image_url, caption=breed_info)
        #await message.answer_photo(stream_image, caption=breed)
        
    else:
        output_image = Image.fromarray(out)
        stream_image = BytesIO()
        stream_image.name = "Dogs.jpg"
        output_image.save(stream_image, 'JPEG')
        stream_image.seek(0)
        # await was here
        await message.answer_photo(stream_image, caption='Выбери породу какого песика ты хочешь узнать)')
        await breed_answering(message, Dogs)
    @dp.message_handler()
    async def test(message1):
        print(message1.text)

def breed_answering(message: types.Message, Dogs):
    for breed in Dogs:
        await message.answer("Что-то не припомню такой породы") 

# dog breed answer
#@dp.message_handler()
async def echo(message: types.Message):
    breed = message.text.lower().title()
    breed_info, image_url = get_breed_info(breed)
    if breed_info == " ":
        await message.answer("Что-то не припомню такой породы")
    await message.answer_photo(image_url, caption=breed_info)


# run long-polling
if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
