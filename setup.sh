wget https://files.pushshift.io/twitter/US_PoliticalTweets.tar.gz
tar xfz US_PoliticalTweets.tar.gz

gunzip tweets/users_final.json.gz
mv tweets/users_final.json .

mv tweets.json tweets_full.json
head -n 100000 tweets_full.json > tweets.json
