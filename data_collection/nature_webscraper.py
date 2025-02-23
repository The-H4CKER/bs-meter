import requests
from bs4 import BeautifulSoup
from selenium import  webdriver
import os

def get_article(suffix):
        r = requests.get('https://www.nature.com' + suffix)
        if r.status_code == 200:
            return r
        else:
            print('invalid get request')


def get_article_body(r):
    soup = BeautifulSoup(r.content, 'html.parser')
    title = soup.find('h1', class_ = 'c-article-title')
    s = soup.find('div', class_='main-content')
    content = soup.find_all('p')
    with open(os.path.join("nature_climate_articles", title.getText() + ".txt"), "w", encoding="utf-8") as f:
        f.write(s.getText(separator=' ', strip=True))



# Making a GET request
r = requests.get('https://www.nature.com/nclimate/research-articles')
# check status code for response received
# success code - 200
print(r)

soup = BeautifulSoup(r.content, 'html.parser')
s = soup.find_all('a', class_='c-card__link u-link-inherit')
for elt in s:
    suffix = elt['href']
    print(suffix)
    response = get_article(suffix)
    get_article_body(response)


'''
# Parsing the HTML
with open("htmlresponse.txt", "w", encoding="utf-8") as f:
    f.write(s.getText(separator=' ', strip=True))

driver = webdriver.Firefox()
driver.get('https://www.nature.com/nclimate/research-articles')
print(driver.title)
'''
