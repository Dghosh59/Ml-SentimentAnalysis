import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from Contractions import contractions
from textblob import TextBlob
from bs4 import BeautifulSoup
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context



nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('vader_lexicon')


class TextPreproccessor:

    def get_wordnet_pos(self,tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # default to noun

    # Lemmatization function
    def lemmatize_text(self,text : str):
        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        lemmatized_words = [
            lemmatizer.lemmatize(token, self.get_wordnet_pos(pos))
            for token, pos in pos_tags
        ]
        return ' '.join(lemmatized_words)

    def remove_punctuation(self,text : str):

        text = text.translate(str.maketrans("", "", string.punctuation))

        return text
    
    def remove_html_tags(self, text: str) -> str:
        """Removes HTML tags from a string using BeautifulSoup."""
        # Use the 'lxml' parser for speed and efficiency
        soup = BeautifulSoup(text, "lxml")
        return soup.get_text()
    
    def remove_stopwords(self,text,reserved = {"not", "no", "never", "none","neither","nor"}):
        stop_words = set(stopwords.words("english"))
        words = [word for word in text.split() if word not in stop_words or word in reserved]
        return " ".join(words)

    def handle_negation(self,text):
        neg_words = {"not", "no", "never", "none", "hardly", "scarcely", "barely","neither","nor"}
        words = text.split()
        new_words = []
        negate = False

        for w in words:
            lw = w.lower()
            if lw in neg_words:
                negate = True
                #new_words.append(lw)
            elif re.search(r'[.!?,;]', w):  # end of sentence or phrase
                negate = False
                new_words.append(w)
            elif negate:
                new_words.append("not" + w)
                negate = False
            else:
                new_words.append(w)
        return " ".join(new_words)
    

    def clean_contractions(self,text : str):

        updated_contractions = {k.replace("'", ""): v for k, v in contractions.items()}
        
        words = [updated_contractions[word] if word in updated_contractions else word for word in text.split()]

        return " ".join(words)


    def stemming(self,text : str):

        stemmer = PorterStemmer()
        
        words = [stemmer.stem(word) for word in text.split()]

        return " ".join(words)

    def clean_numbers(self,text: str) -> str:
        text = re.sub(r'\d+', '', text)   
        text = re.sub(r'\s+', ' ', text).strip() 
        return text

    def correct_text(self,text : str):

        text = str(TextBlob(text).correct())
        return text


    def preproccess(self,text: str ):

        text = text.lower().strip()
        text = self.remove_html_tags(text)
        text = self.remove_punctuation(text)
        text = self.clean_contractions(text)
        text = self.remove_stopwords(text)
        text = self.handle_negation(text)
        text = self.clean_numbers(text)
        text = self.lemmatize_text(text)
        #text = self.remove_stopwords(text)

        return text

        
    def do_prepreoccessing(self,df : pd.DataFrame,column_name : str):
        return df[column_name].apply(self.preproccess)

# if __name__=="__main__":
#     preproccessor = TextPreproccessor() 
#     print(preproccessor.preproccess("Unique and luxurious to be sure. I couldn't recommend staying with Derk any more highly.<br/><br/>Coolest thing first: We showed up to Derk's and he actually had three houseboats in a row, and offered to show up the one we saw on the site and another, and let us choose our favorite to stay in!  We had an incredible boat with wired internet, a comfortable bed and a very comfortable futon, TV, speakers, hot water, a shower, washer/dryer.  I couldn't believe I was on a boat.  Plus the charm of being on one was fantastic, and the views out onto the canal were fantastic.  It rained one morning and was such a beautiful way to wake up.<br/><br/>In addition, Derk was really communicative, responded to our last minute (day before!) request for a reservation quite casually, and met us right at the boat.  He even had a few cold beers in the fridge for us, and was really great explaining the area.<br/><br/>Would recommend!"))