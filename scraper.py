"""
Scrape the course description from "Class Schedule and Quota"
* Column: 15 attributes (See Comments* below)
* Row: 2636 courses. Note that same courses offered in different semesters are considered different in the table (as some of their data are different as well.)
* Time: 4 semesters. From Winter 2017-18 to Fall 2018-19
* Format: .csv (follows the excel dialect)
* Subject: all UG and PG courses offered in the past one year
"""
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import csv
import numpy as np

ats_list = [
    "Subject",        # COMP
    "Catalog",        # 4971C
    "Long Title",     # Independent Work
    "Unit",           # 3
    "Term",           # 1840 means 2018-19 Fall
    
    # Common
    "DESCRIPTION",
    "PREVIOUSCODE",
    
    # UG
    "ATTRIBUTES",
    "EXCLUSION",
    "PRE-REQUISITE",
    "CO-REQUISITE",
    "CO-LISTWITH",
    "ALTERNATECODE(S)",
    
    # PG
    "VECTOR",
    "INTENDEDLEARNINGOUTCOMES"

    # To be implemented
    #instr = dict()
]

ats = dict(zip(ats_list,range(len(ats_list))))


def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 
            and content_type is not None 
            and content_type.find('html') > -1)


def log_error(e):
    """
    It is always a good idea to log errors. 
    This function just prints them, but you can
    make it do anything.
    """
    print(e)


def get_html(term, dept=""):
    url = 'https://w5.ab.ust.hk/wcq/cgi-bin/' + term + '/'
    if dept!="":
        url += 'subject/' + dept
    
    raw_html = simple_get(url)
    return BeautifulSoup(raw_html, 'html.parser')

# debugging helper function
def _checkStr(s):
    print('|'+s+'|')

def split_title(title):
    s = title.split(' - ',1)
    s1 = s[0].split(' ',1)
    s2 = s[1].split(' (')
    if len(s2)>2:
        s2[0] = s2[0]+s2[1]
        s2[1] = s2[2]
    subject = s1[0]
    catalog = s1[1]
    longTitle = s2[0]
    unit = s2[1][0]
    return (subject, catalog, longTitle, unit)


if __name__ == "__main__":
    with open('test.csv', 'a', newline='\n') as csvfile:
        writer = csv.writer(csvfile, dialect='excel')
                            
        writer.writerow(ats_list)
        
        terms = ['1720', '1730', '1740', '1810']
        for term in terms:
            print(term)
            try:
                init_html = get_html(term)
                d = init_html.select('div.depts')[0].text
                depts = list(d[4*i:4*i+4] for i in range(int(len(d)/4)))
                for dept in depts:
                    print(" "+dept)
                    html = get_html(term, dept)

                    for chtml in html.select('div.course'):
                        course = ["" for i in range(len(ats))]
                        
                        tts = split_title( chtml.select('h2')[0].text )
                        course[ ats["Subject"] ]    = tts[0]
                        course[ ats["Catalog"] ]    = tts[1]
                        course[ ats["Long Title"] ] = tts[2]
                        course[ ats["Unit"] ]       = tts[3]
                        course[ ats['Term'] ]       = term

                        cpopup = chtml.select('div.popup.courseattr')
                        if len(cpopup)>0: 
                            cinfos = cpopup[0].select('tr')
                            if len(cinfos)>0:
                                for cinfo in cinfos:
                                    if len(cinfo.select('th'))>0 and len(cinfo.select('td')[0])>0:
                                        key = cinfo.select('th')[0].text.replace(' ','')
                                        val = cinfo.select('td')[0].text
                                        course[ ats[key] ] = val 
                        
                        writer.writerow(course)

            except Exception as e:
                log_error(e)
        
        """
        try:
            html = get_html('1730', 'ACCT')

            for chtml in html.select('div.course'):
                course = ["" for i in range(len(ats))]
                
                print(chtml.select('h2')[0].text )
                tts = split_title( chtml.select('h2')[0].text )
                course[ ats["Subject"] ]    = tts[0]
                course[ ats["Catalog"] ]    = tts[1]
                course[ ats["Long Title"] ] = tts[2]
                course[ ats["Unit"] ]       = tts[3]
                course[ ats['Term'] ]       = '1740'

                cpopup = chtml.select('div.popup.courseattr')
                if len(cpopup)>0: 
                    cinfos = cpopup[0].select('tr')
                    if len(cinfos)>0:
                        for cinfo in cinfos:
                            if len(cinfo.select('th'))>0 and len(cinfo.select('td')[0])>0:
                                key = cinfo.select('th')[0].text.replace(' ','')
                                val = cinfo.select('td')[0].text
                                course[ ats[key] ] = val 

        except Exception as e:
            log_error(e)
        """
                