import requests
from bs4 import BeautifulSoup
from selenium import  webdriver

def get_article(suffix):
        r = requests.get('https://www.nature.com' + suffix)
        if r.status_code == 200:
            return r
        else:
            print('invalid get request')


def get_article_body(reponse):
    soup = BeautifulSoup(response.content, 'html.parser')
    s = soup.find('div', class_='main-body')
    print(s.get_text(separator=' ', strip=True))


# Making a GET request
r = requests.get('https://www.nature.com/nclimate/research-articles')
# check status code for response received
# success code - 200
print(r)

soup = BeautifulSoup(r.content, 'html.parser')
s = soup.find_all('a', class_='c-card__link u-link-inherit')
for elt in s:
    suffix = elt['href']
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
