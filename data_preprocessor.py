import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import gazetteers
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TweetFilter(object):

    def __init__(self):
        self.gazetteers = [x.lower() for x in gazetteers.words()]
        self.stopwords  = [x.lower() for x in stopwords.words('english')]

        self.rx_space   = r'\s+'
        self.rx_email   = r'[a-zA-Z0-9+_\-\.]+@[0-9a-zA-Z][.-0-9a-zA-Z]*.[a-zA-Z]+'
        self.rx_url     = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        self.rx_hashtag = r'#(\w+)'
        self.rx_mention = r'@(\w+)'
        self.rx_empty   = "empty"

    def clean_tweet(self, tweet):
        tweet = re.sub(self.rx_email, 'email', tweet)
        tweet = re.sub(self.rx_url, 'url', tweet)
        tweet = re.sub(self.rx_mention, '', tweet)
        tweet = re.sub(self.rx_space, ' ', tweet)
        return tweet

    def get_tokens(self, tweet):
        tweet = tweet.strip().lower()
        tweet = self.clean_tweet(tweet)
        tokens = TweetTokenizer().tokenize(tweet)
        #tokens = [t for t in tokens if not t in self.stopwords]
        #tokens = [t for t in tokens if not t in self.gazetteers]
        #tokens = [WordNetLemmatizer().lemmatize(t) for t in tokens]
        if len(tokens) == 0: tokens = [self.rx_empty]
        return tokens

if __name__ == '__main__':
    sample_tweets = [
        "This is first line!",
        "2nd line e e e..."
    ]

    tweet_tokens = []
    for t in sample_tweets:
        tokens = TweetFilter().get_tokens(t)
        tweet_tokens.append(tokens)

    print(sample_tweets[0])
    print(tweet_tokens[0])
