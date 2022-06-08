from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import re
import time

def crawl_data():
    try:
        department = driver.find_element_by_xpath(f'//*[@id="au_board_list"]/tr[{i}]/td[2]/a').text # 과 찾아서 text 저장
        department = re.compile('[^가-힣a-zA-Z ]').sub(' ', department) # 필요한 것만 가지기
        time.sleep(0.01)
        driver.find_element_by_xpath(f'//*[@id="au_board_list"]/tr[{i}]/td[1]/a').click() # 해당하는 글 클릭
        time.sleep(0.01)
        title = driver.find_element_by_xpath('//*[@id="content"]/div[1]/div/div[1]/div[2]/div/div').text # 제목 찾아서 text 저장
        title = re.compile('[^가-힣a-zA-Z ]').sub(' ', title) # 필요한 것만 가지기
        content = driver.find_element_by_xpath('//*[@id="content"]/div[1]/div/div[1]/div[3]').text # 글 찾아서 text 저장
        content = re.compile('[^가-힣a-zA-Z ]').sub(' ', content)# 필요한 것만 가지기
        title_list.append(title) # 만든 list에 추가
        content_list.append(content)
        department_list.append(department)
        driver.get(url) # 한 번 클릭해서 들어 왔기 때문에 url을 다시 지정해 줘야한다
    except NoSuchElementException:
        print('NoSuchElementException')
        driver.get(url)
        # 주로 제목이 없거나, 글이 없는 경우 except 가 발생-> 시간을 많이 잡아 먹는다 대략 7~8초 정도?

option = webdriver.ChromeOptions()
# options.add_argument('headless')  이거 활성화하면 웹 브라우저가 안 뜸. 보고싶으면 주석 풀면 되고. 근데지금은 주석하래. 지금은 이거 하면 에러뜬대

option.add_argument('lang=ko_KR')
option.add_argument('--no-sandbox')   #이 아래 3개는 맥 어쩌고에서 필요한 거임. 윈도우 주석 풀어놔도 괜찮음.
option.add_argument('--disable-dev-shm-usage')
option.add_argument('disable-gpu')
option.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36")
driver= webdriver.Chrome('./chromedriver', options = option)
driver.implicitly_wait(10)


df_data = pd.DataFrame()

for l in range(8,9): # 의료과 입력 1:내과 27:핵의학과
    title_list = [] # 우리가 쓸 요소 리스트 만들기
    content_list = []
    department_list = []
    for k in range(1, 101): # 페이지 range 1 page ~ 500 page
        # url = f'https://kin.naver.com/qna/expertAnswerList.naver?dirId=701{l}&queryTime=2022-04-01%2011%3A39%3A23&page={k}' # l =과 , k = page, 큰 과 사용시
        url = f'https://kin.naver.com/qna/expertAnswerList.naver?dirId=701010{l}&queryTime=2022-04-01%2014%3A14%3A29&page={k}' # l =과 , k = page ,세부 과 사용시
        driver.get(url) #url앞의 url 받기
        time.sleep(0.01)
        for i in range(1, 21): # page 안의 글, 총 20개
            try:
                crawl_data()
            except StaleElementReferenceException:
                driver.get(url)
                time.sleep(0.01)
                crawl_data()
            except:
                print('error')
        if k % 10 == 0:
            df_section_title = pd.DataFrame(title_list, columns=['title'])  # 리스트를 dataFrame화
            df_section_content = pd.DataFrame(content_list, columns=['content'])  # 리스트를 dataFrame화
            df_section_department = pd.DataFrame(department_list, columns=['department'])  # 리스트를 dataFrame화
            df_data = pd.concat([df_data, df_section_title, df_section_content, df_section_department],
                                axis='columns', ignore_index=True)  # 만든 dataFrame 합치기, axis='columns'->옆으로 합치기
            df_data.to_csv('./crawling_data/medical_qs_0108_Rheumatology{}.csv'.format(k), index=False)  # index=False-> 만든 csv에 index 제거
            title_list = []  # 우리가 쓸 요소 리스트 만들기
            content_list = []
            department_list = []
            df_data = pd.DataFrame()


driver.close()
df_data.info()
