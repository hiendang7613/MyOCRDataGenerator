from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import os
import random

url = 'https://www.bing.com/images/search?view=detailV2&insightstoken=bcid_ry1oQ-kLJg4H.w*ccid_LWhD6Qsm&form=SBIHMP&iss=SBIUPLOADGET&sbisrc=ImgPicker&idpbck=1&sbifsz=1840+x+920+%c2%b7+27.69+kB+%c2%b7+png&sbifnm=Screenshot+2024-05-23+at+09.52.42.png&thw=1840&thh=920&ptime=33&dlen=37808&expw=848&exph=424&selectedindex=0&id=194385660&ccid=LWhD6Qsm&vt=2&sim=1'

options = Options()
# options.add_argument("--headless=new")  # Enable headless mode
# options.add_argument('--disable-gpu')
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36")

driver = webdriver.Chrome(options=options)
driver.get(url)
driver.implicitly_wait(2)

#read file image_sources_old.txt to list
if os.path.exists('/Users/apple/selenium1/image_sources_old_X.txt'):
    with open('/Users/apple/selenium1/image_sources_old_X.txt', 'r') as file:
        img_src_list = file.readlines()
    img_src_list = set([img_src[:-1] for img_src in img_src_list])
else:
    img_src_list = set()

#read file image_sources_old.txt to list
if os.path.exists('old_related_urls.txt'):
    with open('old_related_urls.txt', 'r') as file:
        old_related_urls = file.readlines()
    old_related_urls = set([img_src[:-1] for img_src in old_related_urls])
else:
    old_related_urls = set()


new_img_src_list = set()

related_urls = set()
old_num=len(img_src_list)

while True:

    while True:
        try:
            cur_related_urls = set([element.get_attribute('href') for element in driver.find_elements(By.CSS_SELECTOR, '.richImgLnk')])

            img_src = driver.find_element(By.CSS_SELECTOR, '.mainImage.current img').get_attribute('src')
            if img_src not in img_src_list:
                img_src_list.add(img_src)
                new_img_src_list.add(img_src)
                if len(new_img_src_list) % 10==0 :
                    print(f"Number of images: {len(new_img_src_list)}")
                    with open('image_sources.txt', 'w') as file:
                        for last_img_src in new_img_src_list:
                            file.write(f"{last_img_src}\n")
            next_button = driver.find_element(By.CSS_SELECTOR, '#navr')
            next_button.click()
        except:
            break
    cur_related_urls.difference_update(old_related_urls)
    related_urls.update(cur_related_urls)
    
    while True:
        # url = related_urls.pop()
        url = random.sample(related_urls, 1)[0]
        related_urls.remove(url)

        old_related_urls.add(url)
        # Write img_src_list to a file
        with open('old_related_urls.txt', 'a') as file:
            for old_related_url in old_related_urls:
                file.write(f"{old_related_url}\n")
        driver.get(url)
        try:
            img_src = driver.find_element(By.CSS_SELECTOR, '.mainImage.current img').get_attribute('src')
        except:
            continue
        if img_src not in img_src_list:
            break
        print(f"Found duplicate: {img_src}")

    # driver.implicitly_wait(2)


driver.quit()
