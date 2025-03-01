import requests
from bs4 import BeautifulSoup
import os
import re

def get_article(link):
        r = requests.get(link)
        if r.status_code == 200:
            return r
        else:
            print('invalid get request')


def get_article_body(r):
    soup = BeautifulSoup(r.content, 'html.parser')
    title = soup.find('h1', class_ = 'c-article-title')
    clean = re.sub(r'[^a-zA-Z0-9]', '', title.getText())
    s = soup.find('div', class_='main-content')
    with open(os.path.join("climate_policy_articles", clean[:50] + ".txt"), "w", encoding="utf-8") as f:
        f.write(s.getText(separator=' ', strip=True))


# Making a GET request
r = requests.get('https://link.springer.com/journal/10584/articles')
# check status code for response received
# success code - 200

soup = BeautifulSoup(r.content, 'html.parser')
s = soup.find_all('a', attrs={"data-track": "select_article"})
for elt in s:
    link = elt['href']
    response = get_article(link)
    print(response)
    get_article_body(response)
