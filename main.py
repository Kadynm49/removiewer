import requests
import sys
try:
    import simplejson as json
except ImportError:
    import json
from bs4 import BeautifulSoup
from bs4 import NavigableString
from bs4 import Tag
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.corpus import reuters
from nltk import bigrams, trigrams
from collections import Counter, defaultdict
import random
from twitter import Twitter, OAuth

MAX_TWEET_LENGTH = 280
MARKOV   = "markov"
TRIGRAM  = "trigram"
POSITIVE = "positive"
AVERAGE  = "average"
NEGATIVE = "negative"

def generate_review(title, review_type, model):

    # Find imdb title key for this movie using the omdb API
    omdb_key = "a542cac9"
    link = "http://www.omdbapi.com/?apikey=" + omdb_key
    t = "&t=" + title
    response = requests.get(link + t)
    if response.status_code != 200:
        print("OMDBAPI is down, sorry! :'(")
        exit()
    responseJson = json.loads(response.text)
    imdbID = ""
    if "imdbID" in responseJson:
        imdbID = responseJson["imdbID"]
        title = responseJson["Title"]
    else:
        raise Exception("Invalid title, please try a different movie or show.")

    # Get imdb user reviews page
    URL = "https://www.imdb.com/title/" + imdbID + "/reviews/"
    load_more_url = "https://www.imdb.com/title/" + imdbID + "/reviews/_ajax?paginationKey="
    r = requests.get(url = URL)
    soup = BeautifulSoup(r.text, "html.parser")

    # Remove br, ul, ol, li, a tags
    for e in soup.find_all("br"):
        e.extract()
    for e in soup.find_all("ul"):
        e.extract()
    for e in soup.find_all("ol"):
        e.extract()
    for e in soup.find_all("li"):
        e.extract()
    for e in soup.find_all("a"):
        e.extract()

    # Find load more button
    load_more_button = soup.find(class_ = "load-more-data")

    reviews = []
    review_titles = []
    avg_score = 0

    negative_reviews = []
    negative_review_titles = []
    avg_negative_score = 0

    positive_reviews = []
    positive_review_titles = []
    avg_positive_score = 0

    # Get reviews by loading all pages of user reviews
    # IMDB uses a "Load More" button to retrieve 25
    # reviews at a time, so we must find the paginationKey
    # for each page to get the next 25 reviews and append
    # those reviews to the list. Stop when there is no
    # longer a "Load More" button

    while (load_more_button != None):
        review_containers = soup.find_all("div", class_="lister-item-content")
        for review_container in review_containers:
            # Get review score
            score = None
            for content in review_container.contents:
                if type(content) == Tag and content["class"][0] == "ipl-ratings-bar":
                    score = int(list(filter(lambda x: type(x) == Tag and x["class"][0] == "rating-other-user-rating", content.contents))[0].contents[3].contents[0])
                    if (score >= 5):
                        avg_positive_score += score
                    else:
                        avg_negative_score += score
                    avg_score += score

            # Get review and review's title
            for content in review_container.contents:
                if type(content) == Tag \
                    and content["class"][0] == "content" \
                    and len(content.contents) > 0 \
                    and len(content.contents[1].contents) > 0 \
                    and type(content.contents[1].contents[0]) == NavigableString: # failed on this line for Avengers: Endgame

                    review = None
                    if len(content.contents[1].contents) == 1:
                        review = content.contents[1].contents[0]
                    else:
                        review = " ".join(content.contents[1].contents)

                    if (score is not None and score >= 5):
                        positive_reviews.append(review)
                    elif (score is not None):
                        negative_reviews.append(review)
                    reviews.append(review)

                if type(content) == Tag and content["class"][0] == "title":
                    if score is not None and score >= 5:
                        positive_review_titles.append(content.contents[0])
                    elif score is not None:
                        negative_review_titles.append(content.contents[0])
                    review_titles.append(content.contents[0])

        if ("data-key" not in load_more_button.attrs):
            if (len(reviews) == 0):
                raise Exception("No reviews for this movie: " + URL)
            else:
                break

        # Get url for next 25 reviews and load them in
        URL = load_more_url + load_more_button["data-key"]
        r = requests.get(url = URL)
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove unnecessary tags
        for e in soup.find_all("br"):
            e.extract()
        for e in soup.find_all("ul"):
            e.extract()
        for e in soup.find_all("ol"):
            e.extract()
        for e in soup.find_all("li"):
            e.extract()
        for e in soup.find_all("a"):
            e.extract()

        # Find next load more button
        load_more_button = soup.find(class_="load-more-data")

    if (len(positive_reviews) == 0 and review_type == POSITIVE):
        raise Exception("No positive reviews exist for " + title)
    elif (review_type == POSITIVE):
        avg_positive_score /= len(positive_reviews)

    if (len(negative_reviews) == 0 and review_type == NEGATIVE):
        raise Exception("No negative reviews exist for " + title)
    elif (review_type == NEGATIVE):
        avg_negative_score /= len(negative_reviews)

    avg_score /= len(reviews)

    # Calculate average review length
    avg_review_len = 0
    avg_positive_review_len = 0
    avg_negative_review_len = 0
    for review in reviews:
        review_len = len(review.split())
        avg_review_len += review_len

        if review in positive_reviews:
            avg_positive_review_len += review_len
        elif review in negative_reviews:
            avg_negative_review_len += review_len
    avg_review_len //= len(reviews)

    if (review_type == NEGATIVE or review_type == POSITIVE):
        avg_positive_review_len //= len(positive_reviews)
        avg_negative_review_len //= len(negative_reviews)

    # All reviews in one string
    reviews_string = " ".join(reviews)
    positive_reviews_string = " ".join(positive_reviews)
    negative_reviews_string = " ".join(negative_reviews)

    if (model == TRIGRAM):
        if review_type == AVERAGE:
            return create_review_with_trigrams(reviews_string.split(), avg_review_len, title, avg_score), imdbID
        elif review_type == POSITIVE:
            return create_review_with_trigrams(positive_reviews_string.split(), avg_positive_review_len, title, avg_positive_score), imdbID
        elif review_type == NEGATIVE:
            return create_review_with_trigrams(negative_reviews_string.split(), avg_negative_review_len, title, avg_negative_score), imdbID
        else:
            print(review_type, "is not a valid option for review type.")
    else:
        if review_type == AVERAGE:
            try:
                return create_review_with_markov_chains(reviews_string.split(), avg_review_len, title, avg_score), imdbID
            except:
                raise
        elif review_type == POSITIVE:
            try:
                return create_review_with_markov_chains(positive_reviews_string.split(), avg_positive_review_len, title, avg_positive_score), imdbID
            except:
                raise
        elif review_type == NEGATIVE:
            try:
                return create_review_with_markov_chains(negative_reviews_string.split(), avg_negative_review_len, title, avg_negative_score), imdbID
            except:
                raise

# markov chain method
# Source: https://www.jeffcarp.com/posts/2019/markov-chain-python/
def create_review_with_markov_chains(corpus, review_len, title, avg_score):
    rating = " %.1f/10\n" % avg_score
    # Create graph
    markov_graph = defaultdict(lambda: defaultdict(int))

    last_word = corpus[0]
    for word in corpus[1:]:
        markov_graph[last_word][word] += 1
        last_word = word

    try:
        return rating + ' '.join(walk_graph(markov_graph, distance=review_len))
    except:
        raise

# Source: https://www.jeffcarp.com/posts/2019/markov-chain-python/
def walk_graph(graph, distance, start_node=None):

    # If not given, pick a start node at random.
    if not start_node:
        start_node = random.choice(list(graph.keys()))
        while start_node.islower():
            start_node = np.random.choice(list(graph.keys()))

    weights = np.array(
        list(graph[start_node].values()),
        dtype=np.float64)
    # Normalize word counts to sum to 1.
    weights /= weights.sum()

    # Pick a destination using weighted distribution.
    choices = list(graph[start_node].keys())
    try:
        chosen_word = np.random.choice(choices, None, p=weights)
    except:
        raise

    if distance <= 0 and chosen_word[-1] == ".":
        return [chosen_word]

    return [chosen_word] + walk_graph(graph, distance=distance-1, start_node=chosen_word)

# trigram method
# created with @zadelacerda in a college NLP course
def create_review_with_trigrams(corpus, review_len, title, avg_score):
    rating = " %.1f/10\n" % avg_score

    trigrams = list(ngrams(corpus, 3))
    unsmoothed_trigrams = Counter(ngrams(corpus, 3))

    trigram_count = len(unsmoothed_trigrams)

    for trigram in unsmoothed_trigrams:
        unsmoothed_trigrams[trigram] /= trigram_count

    randint = np.random.randint(trigram_count)

    while trigrams[randint][2].islower():
        randint = np.random.randint(trigram_count)

    first_word = (trigrams[randint], unsmoothed_trigrams[trigrams[randint]])

    review = ""
    # Generate sentences using unsmoothed trigram model
    word_count = 0
    while True:
        randint = np.random.randint(trigram_count)

        while trigrams[randint][2].islower():
            randint = np.random.randint(trigram_count)

        first_word = (trigrams[randint], unsmoothed_trigrams[trigrams[randint]])

        current_word = first_word
        sentence = ""
        while current_word[0][2][-1] != "." and word_count <= review_len:
            sentence += current_word[0][2] + " "
            next_word = { "prob" : 0, "word": "" }
            next_word_candidates = []
            # Find most likely word to come after current word
            for trigram in unsmoothed_trigrams:
                # Set most likely word if this trigram has the highest probability found so far
                if trigram[1] == current_word[0][2] and trigram[0] == current_word[0][1] and unsmoothed_trigrams[trigram] > next_word["prob"]:
                    next_word["prob"] = unsmoothed_trigrams[trigram]
                    next_word_candidates.append({"prob": unsmoothed_trigrams[trigram], "word":trigram})

            next_word_candidates = sorted(next_word_candidates, key = lambda i: i["prob"])
            if (len(next_word_candidates) >= 3):
                next_word = next_word_candidates[random.randrange(0, 3)]
            else:
                next_word = next_word_candidates[random.randrange(0, len(next_word_candidates))]
            current_word = (next_word["word"], next_word["prob"])

            word_count += 1
        sentence += current_word[0][2] + " "
        review += sentence

        if word_count > review_len:
            break

    return rating +  review

# Use randomlists.com to get a random movie title to review
def random_movie_title():
    URL = "https://www.randomlists.com/random-movies?dup=false&qty=1"
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    options = Options()
    options.headless = True
    driver = webdriver.Firefox(options=options)
    driver.get(URL)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "rand_medium"))
    )
    title = driver.find_element_by_class_name("rand_medium").text
    driver.quit()
    return title

# Get a movie title that has not already been reviewed
def watch_new_movie():
    title = random_movie_title()

    while True:
        reviewed_titles = open("/home/erviewre/twitter-movie-reviewer/review_log.csv", "r")
        for line in reviewed_titles:
            if (line[:line.index(",")].strip() == title):
                title = random_movie_title()
                reviewed_titles.close()
                continue
        break

    return title

def post_tweet(review, imdbID):
    review = "https://www.imdb.com/title/" + imdbID + review
    review = trim_to_tweet_size(review)

    if (len(review) <= 50):
        print(len(review))
        return False

    review = add_hashtag(review)

    # Api request setup
    with open('/home/erviewre/twitter-movie-reviewer/config.json') as f:
        config = json.load(f)

    t = Twitter(auth=OAuth(config["accessToken"], config["accessTokenSecret"], config["clientKey"], config["clientSecret"]))
    t.statuses.update(status=review)
    return True

def trim_to_tweet_size(review):
    while len(review) >= MAX_TWEET_LENGTH: # greater than or equal to so that there is room for the hashtag
        try:
            review = review[:review.rindex('.', 0, -1)+1]
        except:
            try:
                review = review[:review.rindex('!', 0, -1)+1]
            except:
                try:
                    review = review[:review.rindex('?', 0, -1)+1]
                except:
                    review = review[0:-1]

    return review

def log_review(title, model, review_type):
    reviewed_titles = open("/home/erviewre/twitter-movie-reviewer/review_log.csv", "a")
    reviewed_titles.write(title + "," + model + "," + review_type + "\n")
    reviewed_titles.close()

def add_hashtag(review):
    word_list = review.split()
    random_word = random.choice(word_list[3:])
    return review.replace(random_word, "#" + random_word, 1)

if __name__ == "__main__":
    title = None
    model = None
    review_type = None
    try:
        title = watch_new_movie()
        model = random.choice([TRIGRAM, TRIGRAM, TRIGRAM, TRIGRAM, TRIGRAM, MARKOV])
        review_type = random.choice([AVERAGE, AVERAGE, AVERAGE, AVERAGE, AVERAGE, NEGATIVE, POSITIVE])
    except Exception as error:
        print(str(error))

    for x in range(3):
        try:
            review, imdbID = generate_review(title, review_type, model)
            if (not post_tweet(review, imdbID)):
                continue
            log_review(title, model, review_type)
            break
        except Exception as error:
            print(str(error))
            if (x == 2):
                review, imdbID = generate_review(title, AVERAGE, TRIGRAM)
                post_tweet(review, imdbID)
                log_review(title, model, review_type)
