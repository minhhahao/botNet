# tob

A chatbot that will cheer will up when you are sad

1.  Processing Database for the bot

-   using Reddit comments from [pushshift.io](https://files.pushshift.io/reddit/comments/)

## Woking in progress

-   [x] Processing RAW JSON file
-   [x] Finish the database
-   [ ] Setting up training dataset
-   [ ] Creating training model

## Logs

### Week 1 - 2 : Creating database

-   In order to create a chatbot, I will need to build a training database for the chatbot, I decide to use Reddit data dump from [pushshift.io](https://files.pushshift.io/reddit/comments/). Current using _2010-10_
-   The format of a RC.json file looks like:
```
  {
    __"parent_id":"t1_c110jjw",__
    __"created_utc":"1285891201",__
    "author":"hbetx9",
    __"subreddit_id":"t5_2qh6p",__
    "score_hidden":false,
    "distinguished":null,
    "subreddit":"Conservative",
    __"score":3,__
    __"body":"Can you give a reference for this. I don't believe that is true, i.e., constitution limits how congress can tax.",__
    "archived":true,
    __"name":"t1_c1112h4",__
    "link_id":"t3_dl2dp",
    "gilded":0,
    "edited":false,
    "ups":3,
    "retrieved_on":1426494582,
    "author_flair_css_class":null,
    "author_flair_text":null,
    "id":"c1112h4",
    "downs":0,
    "controversiality":0
  }
  ```
  
  - One can see there is a lot of unecessary information within this file, and the only crucial information is highlighted above. Because we are procesing through thousands of data lines, it would be time-expensive to add unecessary information (_O(n)_) ==> Firgure out how to remove unecessary dataset

  _Problem 1_: Working with JSON file.
