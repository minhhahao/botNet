# Logs

## Week 1 - 2 : Creating database

-   In order to create a chatbot, I will need to build a training database for the chatbot, I decide to use Reddit data dump from [pushshift.io](https://files.pushshift.io/reddit/comments/). Current using _2010-10_
-   The format of a RC.json file looks like:
```JSON
  {
    "parent_id":"t1_c110jjw",
    "created_utc":"1285891201",
    "author":"hbetx9",
    "subreddit_id":"t5_2qh6p",
    "score_hidden":false,
    "distinguished":null,
    "subreddit":"Conservative",
    "score":3,
    "body":"Can you give a reference for this. I don't believe that is true, i.e., constitution limits how congress can tax.",
    "archived":true,
    "name":"t1_c1112h4",
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
  - necessary information : _"parent_id"_, _"created_utc"_,_"subreddit_id"_,_"score"_,_"body"_, _"name"_

  - One can see there is a lot of unecessary information within this file. Because we are procesing through thousands of data lines, it would be time-expensive to add unecessary information (_O(n)_) ==> Firgure out how to remove unecessary dataset

_Problem 1_: Choosing database language
- There are plenty of choices to process the data, including storing the database : _pandas_ library (`pip install pandas`), server-based language such as SQL. Since I only need to purify the data given from the Reddit Comment dump, I will use SQLite3 for the sake of simplicity (no need to use a server-based language like MySQL), suggested from [here](https://www.sqlite.org/different.html) and [Samuel's answer on SO](https://stackoverflow.com/a/4539621)

_Problem 2_: Constructing the database through _SQLite3_
  - _2.1_: the size of data
      - When I download the most recent RC file from [pushshift.io](https://files.pushshift.io/reddit/comments/), the compressed file is already 10GB, the uncompressed file can easily exceed my 16GB of RAM, therefore I need to buffer through the data and only stored necessary info in a _SQLite3_ Database
      - Since a reply has a parent comment, the idea is that we can insert reply into the database, and we can match the two with same _id_, thus we can have rows that have a parent comment and a reply that go with it
      - Since reddit comments are measured through _score_, I will use score as my parameter to show whether the reply is relevant to the question.
  - _2.2_: `COMMIT` in SQLite3
      - in order to insert rows of data into the database, SQL has a function called `COMMIT` that will put data into the database. Since we are working with millions of rows (currently working with RC from Oct-2010), it would be time-consuming to do it one by one. Therefore, creating a _sql_transaction_ will help put all the queries in one group and execute them all, [This](https://www.tutorialspoint.com/sql/sql-transactions.htm) article from TutorialPoint explains clearly about using transaction in SQL
  - _2.3_: Working I/O file in Python
      - In Java, the process of I/O file in Java requires a `FileWriter`->`BufferedReader`->`FileWriter`->`BufferedWriter`. However, from [this](https://www.guru99.com/reading-and-writing-files-in-python.html) article, using `open()` function with more parameter will do the job for I/O files
  - _2.4_: Working with JSON files
      - From [this](https://developer.rhino3d.com/guides/rhinopython/python-xml-json/) I can know how to work with JSON file in Python with `import json` and `json.loads()` to read all the data inside the JSON file.
      - Inside the RC file, some comments often exceed a certain amount of characters and thus reddit uses `\n` as well as `\r` to go down to a new line. This can be easily fixed with a single function that I wrote inside the database.py `formatted()`
  - _2.5_: Parents and comments
      - All the comments initially don't have a parent (either because it is a top comment or the parent isn't in the document). As going through the file, some might has _parent_id_ already. Therefore, I need to create a function that can add the comment to the existing parent, so that I can have pairs of data as training_data for the model. (simple `SELECT FROM` sql command). If I have a _comment_id_ in the database that match another comment's _parent_id_, then I can match it to the parents in the database.
  - _2.6_: Setting parameter and cleaning data
      - as said before, I will use _score_ to filter out comments that is bad, thus I will create `fscore()` function to find the existing pair whether it has the score >= 2 or not.
      - some comments are either too long or getting removed by the mod in the subreddt, therefore I create a function to check whether the comments are in an appropriate length and that the comments aren't removed or deleted.
  - _2.7_: Inserting the data into the Database
      - The problem is that some comments might already have the parents in the database, some don't. Some might have the same parents but higher score.
      _Solution_:
        - `insrcomment` serves where two comments have the same parents. The new comment happens to get accepted and also have a higher score than the existing comment => this function insert then replace the old one with the new one
        - `inspar` serves as inserting the comment into existing Parents
        - `insnopar` servers as inserting comments into the database without existing _parent_

_Problem 3_: creating _training_data_
  - How can the model understand what is the question and what is the appropriate answers to it?
  - Therefore I need to create a .from file (txt file) to a .to file (txt file) using the from the rows of the database so that the model can have a base to train => therefore I create a `training.py` to create training data. Inside the file include simple writing files and some SQl queries

## Week 3: Learning NMT
- I haven't spent a lot of time writing code this past week because I had to work on the English provincial exam. Nonetheless, I worked on NMT (_Neural Machine Translation_) based on [tensorflow's NMT](https://github.com/tensorflow/nmt), a _many-to-many_ implementation of _Recurrent Neural Network_, or often known as _RNN_
- Some of the links that I used during the past week:
  - [Basis of RNN and LSTMs (Long Short Term Memory)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  - [Tensorflow's RNN Tutorial](https://www.tensorflow.org/tutorials/sequences/recurrent)
  - [Papers](docs/RNN_LearningPhraseRepresentation.pdf)
