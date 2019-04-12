# tob
A chatbot that will cheer will up when you are sad

1. Processing Database for the bot
  - using Reddit comments from [pushshift.io](https://files.pushshift.io/reddit/comments/)

## Woking in progress
- [x] Processing RAW JSON file
- [x] Finish the database
- [ ] Setting up training dataset
- [ ] Creating training model

##Logs

Thought process for buffering the data:
First option: uses SQLite3 to process raw JSON file and then input all necessary value such as _parent_id_ _subreddit_id_ _comment_id_ out
and then input them all into pandas dataframe ==> takes too long to process the all the data

Might still uses SQLite3 but might have to think another solution to solve time complexity


Second option: might as well put straight all the data into pandas using _pd.read_json()_ and then process all the trash data from pandas
dataframe but that shit does not work

Third option: mongoDB?? nope

reddit comment have too much trash to clean ==> makes it longer to train
have to process the data first?
