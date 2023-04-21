from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import config
import json
from main import json_f, predict, color

bot = Bot(token=config.BOT_TOKEN, parse_mode="HTML")
dp = Dispatcher(bot)


@dp.message_handler(commands = "start")
async def start(message: types.Message):
    data = {
        f"{message.from_user.id}":[]
    }
    json_f.write(data, "users.json")
    await message.bot.send_message(message.from_user.id, "Добро пожаловать!")


@dp.message_handler(content_types=['text'])
async def handle_photo(message: types.Message):
    data = json_f.read("users.json")
    data[f"{message.from_user.id}"].append({
        f"{message.text}":"color"
    })
    json_f.write(data, "users.json")


@dp.message_handler(content_types=['photo'])
async def handle_photo(message: types.Message):
    photo = message.photo[-1]
    photo_file = await bot.download_file_by_id(photo.file_id)
    fashion = predict.fashion(photo_file, photo)
    await bot.send_message(message.chat.id, text=f"{fashion}")
    photo_file = await bot.download_file_by_id(photo.file_id)
    colour = color.f(photo_file, photo)
    await bot.send_message(message.chat.id, text=f"{colour}")



if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)