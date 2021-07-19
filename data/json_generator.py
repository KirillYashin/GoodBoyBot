import requests
from bs4 import BeautifulSoup
import json

headers = {
    'authority': 'https://www.doge.ru',
    'cache-control': 'max-age=0',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.106 Safari/537.36',
    'sec-fetch-dest': 'document',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-user': '?1',
    'accept-language': 'en-US,en;q=0.9',
}

session = requests.session()
response = session.get('https://doge.ru/poroda', headers=headers)

soup = BeautifulSoup(response.text, 'html.parser')

breeds = soup.find_all('li', class_='breeds-letter-content__item')

links = []
names = []
output_data = open('data.json', 'w', encoding='utf-8')

for breed in breeds:
    name = breed.getText()
    names.append(name[1:-1])
    links.append(breed.find('a').get('href'))

breeds_ru_to_en = {}
with open('breeds_new.txt', 'r', encoding='utf-8') as breeds_file:
    lines = breeds_file.readlines()
    for line in lines:
        breeds_ru_to_en[line[6:-2].split('::')[1]] = str(line[6:-2].split('::')[0])

#print(breeds_ru_to_en)
i = 0
breeds_info = {}

for link in links:
    response = session.get(str(link), headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    cards = soup.find_all('div', class_='breeds-info-card__right-col')
    description = soup.find_all('div', class_='col-12 col-lg-8 single__col-left')
    photos = soup.find_all('div', class_='single-main__slider')
    current_breed = {}

    for current in description:
        text = str(current.find('div', class_='single__desc single__desc--breed-story').getText())
        current_breed['Описание'] = text[1:].strip()

    for photo in photos:
        ph = photo.find('img').get('src')
        current_breed['Ссылка на картинку'] = str(ph)

    for card in cards:
        data = str(card.find('div', class_='breeds-info-card__name').getText().strip())
        value = str(card.find('div', class_='breeds-info-card__desc').getText().strip())
        value = value.replace(',   ', ', ')
        print(data)
        print(value)
        current_breed[data] = value
    current_breed['Название породы'] = str(names[i])
    current_breed['Ссылка на собаку'] = str(link)

    if str(names[i]) in breeds_ru_to_en.keys():
        breeds_info[breeds_ru_to_en.get(str(names[i]))] = current_breed
    i += 1

current_breed = {
    'Описание': 'Эти собаки имеют благородную историю, являясь потомками французских Grand Bleu de Gascogne и тех, многие из которых были завезены в Соединенные Штаты в колониальные времена. Хотя он в первую очередь охотник, Блюетик может быть прекрасной домашней собакой и любит своих людей.',
    'Ссылка на картинку': 'https://i.pinimg.com/736x/7d/55/2c/7d552c5f8d7d9dc9bae3710e299e8c28--bluetick-coonhound-dog-love.jpg',
    'Стоимость': '10-20 тыс. рублей', 'Длительность жизни': 'От 10 до 12 лет', 'Шерсть': 'Мало линяет',
    'Рождаемость': 'От 5 до 8 щенков', 'Рост в холке, см': 'Мальчик - 71 см, девочка - 63 см',
    'Вес, кг': 'Мальчик - 36 кг, девочка - 27 кг', 'Содержание': 'Требуется', 'Назначение': 'Охотничьи, Компаньоны',
    'Название породы': 'Голубая енотовидная борзая',
    'Ссылка на собаку': 'https://www.dog-time.ru/2021/04/01/golubaja-enotovidnaja-borzaja/'}

breeds_info['bluetick'] = current_breed

current_breed = {
    'Описание': 'Порода собак, выведенная в Великобритании и входящая в группу спаниелей. Порода является самой крупной среди всех спаниелей. Название породы происходит от усадьбы Кламбер-парк в Ноттингемпшире (Великобритания). Порода выведена как подружейная и предназначена для охоты в сухопутной местности.',
    'Ссылка на картинку': 'https://doggav.ru/wp-content/uploads/57771560_cr.jpg',
    'Стоимость': '30-70 тыс. рублей', 'Длительность жизни': 'От 12 до 14 лет', 'Шерсть': 'Сильно линяет',
    'Рождаемость': 'От 4 до 6 щенков', 'Рост в холке, см': 'Мальчик - 50 см, девочка - 45 см',
    'Вес, кг': 'Мальчик - 32 кг, девочка - 28 кг', 'Содержание': 'Не требуется', 'Назначение': 'Охотничьи',
    'Название породы': 'Кламбер-спаниель',
    'Ссылка на собаку': 'https://sobaka.wiki/katalog-porod/klamber-spaniel/'}

breeds_info['clumber, clumber spaniel'] = current_breed

current_breed = {
    'Описание': 'В 12 столетии на территорию современной Венгрии прибыли племена мадьяров, вместе с которыми появились и кувасы. Их впоследствии использовали для работы в горных областях, а также в качестве охранной собаки для знати. Кувасы сопровождали короля везде – в спальне и во время охоты. Именно в это время порода приобрела, современные черты и дошла до наших дней в практически неизменном виде.',
    'Ссылка на картинку': 'https://sobaka.wiki/wp-content/uploads/2019/04/20-6.jpg',
    'Стоимость': '50-100 тыс. рублей', 'Длительность жизни': 'От 10 до 12 лет', 'Шерсть': 'Сильно линяет',
    'Рождаемость': 'От 6 до 8 щенков', 'Рост в холке, см': 'Мальчик - 73 см, девочка - 68 см',
    'Вес, кг': 'Мальчик - 55 кг, девочка - 45 кг', 'Содержание': 'Не требуется', 'Назначение': 'Пастушьи',
    'Название породы': 'Венгерский кувас',
    'Ссылка на собаку': 'https://sobaka.wiki/katalog-porod/kuvas-vengerskij/'}

breeds_info['kuvasz'] = current_breed

json.dump(breeds_info, output_data, sort_keys=False, indent=4, ensure_ascii=False, separators=(',', ': '))

