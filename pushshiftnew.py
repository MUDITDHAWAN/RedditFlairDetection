import pandas as pd
import requests
import json
import csv
import time
import datetime

####creating the api as per the stat and end timestamp and the subreddit and
#### getting a list of data object
def getPushshiftData(after, before, sub):
    url = 'https://api.pushshift.io/reddit/search/submission/?size=1000&after='+str(after)+'&before='+str(before)+'&subreddit='+str(sub)
    print(url)
    r = requests.get(url)
    data = json.loads(r.text)
    return data['data']

####from the data object collects the rquired fields and append it to a list
####which would then be made to a pandas Data Frame and uploaded to .csv file,
####data collected is filtered on the basis of the flair of the submission
def collectSubData(subm):
    flag=0
    subData = list() #list to store data points
    title = subm['title']
    url = subm['url']
    try:
        flair = subm['link_flair_text']

    except KeyError:
        flag=1
        flair = "NaN"
    ####flairs to be filtered
    if(flair not in ["AskIndia" , "Non-Political" , "[R]eddiquette" , "Scheduled" , "Photography" , "Science/Technology" , "Politics" , "Business/Finance" , "Policy/Economy" , "Sport" , "Food"]):
        flag=1
    author = subm['author']
    sub_id = subm['id']
    score = subm['score']
    createdtimestamp=subm['created_utc']
    created = datetime.datetime.fromtimestamp(subm['created_utc']) #1520561700.0
    numComms = subm['num_comments']
    permalink = subm['permalink']
    try:
        entire_text=subm['selftext']
    except KeyError:
        entire_text= "NaN"

    if(flag!=1):
        ###appending info about one submission into the list
        subData.append((sub_id,title,url,author,score,createdtimestamp,created,numComms,permalink,flair,entire_text))
        subStats[sub_id] = subData

#Subreddit to query
sub='india'

#before and after dates
before = "1563866717"  ##July 23, 2019 12:55:17 PM
after = "1532330717"   ##July 23, 2018 12:55:17 PM

subCount = 0
subStats = {}

###  call getPushshiftData() for the frst time
data = getPushshiftData( after, before, sub)

### posts gathered as data are now paarsed and then corresponding the date of
### the last collect post the function is call for later dates for collecting
### individual post from the after date up until before date
while len(data) > 0:
    for submission in data:
        collectSubData(submission)
        subCount+=1

    print(len(data))

    ### calculates the date of the last submission
    print(str(datetime.datetime.fromtimestamp(data[-1]['created_utc'])))
    after = data[-1]['created_utc']

    ### Calls getPushshiftData() with the created date of the last submission
    data = getPushshiftData(after, before, sub)

print(len(data))

### function to upload the collected data into a .csv file for further processing

def updateSubs_file():
    ###counts the number of uploads made
    upload_count = 0

    file ="datafinal.csv"
    ### opens the csv file for data to be uploaded
    with open(file, 'w', newline='', encoding='utf-8') as file:
        a = csv.writer(file, delimiter=',')
        ### column headings
        headers = ["sub_id","title","url","author","score","timestamp","created","numComms","permalink","flair","entire_text"]
        ### wrtes a row into the csv file
        a.writerow(headers)
        for sub in subStats:
            if(subStats[sub][0][8]=="Nan"):
                break
            a.writerow(subStats[sub][0])
            upload_count+=1

        print(str(upload_count) + " submissions have been uploaded")

updateSubs_file()
