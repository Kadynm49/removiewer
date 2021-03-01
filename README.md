# removiewer

This project is an extension of this original project: https://github.com/Kadynm49/shrek-cloud/tree/master/final

I made it into a twitter bot 🤖

🎥 https://twitter.com/removiewer 🎥

I have the bot running on an ubuntu VM in Azure, set with the following crontab file:

```
0 15,19,23 * * * python3 /removiewer/main.py >>/home/erviewre/twitter-movie-reviewer/error_log.txt 2>&1
```
