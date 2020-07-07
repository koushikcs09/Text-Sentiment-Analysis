# Text Sentiment Analysis

### Import required libraries
```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd
```
### Loading the Data
The Text (.xlsx) file has been loaded into Pandas Data Frame using the read_excel function.

### Cleaning and Lemmatization
**1.  Remove Unicode:** The Unicode characters `\<\\u[0-9A-Fa-f]+ ^\x00-\x7f> `present in the incoming text needs to be removed.  

**2. Replace URL:** The URL or web link characters `< ((www\.[^\s]+)|(https?://[^\s]+))> `present in the string needs to be cleansed.

**3.  Replace @User:** The text file might have contains `<@user>`, which needs to be removed before processing the file.

**4.  Remove Hash Tag:** The Hash Tag Characters <#> present in the text needs to be removed.

**5.  Remove numbers, special characters, multiple exclamation mark, multiple full stop and multiple commas.**

**6.  Lemmatization :** Once the Text cleansing is done in above steps, the text has been broken into individual words and the lemma (root word) has been identified from the individual words using NLTK wordnet lemmatizer based on the significance of the word (NLTK pos-tag) in the corresponding sentence.    
```
Noun tag => ['NN', 'NNS', 'NNP', 'NNPS']
Verb tag => ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
Adjective tag => ['JJ', 'JJR', 'JJS']
Adverb tag => ['RB', 'RBR', 'RBS']
```
### Python Sentiment Analyzer
1.  Vader Sentiment Analyzer
```python
sid = SentimentIntensityAnalyzer()

text_data['Vader_compound_polarity']=text_data[text_col_name].apply(lambda x:sid.polarity_scores(x)['compound'])

text_data['Vader_sentiment_type']=''

text_data.loc[text_data.Vader_compound_polarity>0,'Vader_sentiment_type']='POSITIVE'

text_data.loc[text_data.Vader_compound_polarity==0,'Vader_sentiment_type']='NEUTRAL'

text_data.loc[text_data.Vader_compound_polarity<0,'Vader_sentiment_type']='NEGATIVE'
```
2. Textblob Sentiment Analyzer
```python
senti = TextBlob(text_data)
polarity = senti.sentiment.polarity
```
### Disadvantages of Sentiment Analysis Process using existing Python Packages (Textblob,Vader) and resolution:
1.  **Limitations of Textblob and Vader:**   

The dictionary maintained by *Textblob* and *Vader* needs to be enhanced to get the accurate sentiment. 

The Text has been scanned as a pre-process to identify the keywords not handled by Textblob and Vader. The customized dictionary has been created with the keywords not handled by Textblob and Vader along with their sentiment type (*Positive/Negative*).
`E.g, ‘Lengthy’ is a keyword not handled by Textblob and Vader.`

`The Process is` **lengthy** (Textblob, Vader **\->** Neutral)
Customized Dictionary for Negative Keywords:` [lengthy,…]`

The idea is to compel textblob or Vader to handle lengthy as negative keyword.

*`Negative Keyword`: Add “bad” as prefix to keyword in sentence.
`Positive Keyword`: Add “good” as prefix to keyword in sentence.*

The keyword “bad” and “good” has been identified by both textblob and Vader as *Positive* and *Negative* sentiment correspondingly.

The Process is **lengthy (Neutral)** **\->** The Process is bad **lengthy (Negative)**

#### Mixed Sentiment: 
The Text might contain mixed sentiment – a portion of the text having positive sentiment and other portion might have negative sense.
`E.g. : I am  `**happy**` with the Agent service but the automated `**system needs to be improved**.

*Resolution:* The Text has been split into multiple phrases to identify the hidden sentiment more accurately.

#### Multiple Adjective:
The text might contain multiple adjective (both positive and negative) to emphasize on a particular thing.

*E.g :* The application is much more slower. <Here, the words positive words - ‘much’, ’more’ puts stress on the negative word ‘slower’. The Mixed combination of adjectives / adverb (positive and negative) might results in different sentiments.

*Resolution:* The Multiple adjective / adverb has been replaced with single main adjective to identify the actual polarity of the text.

In the above example, The application is much more **slower (Mixed)**  \-> The application is **slower (Negative)**

#### Phrases:
The Text might contain Phrases, which needs to convert to meaningful sentence to get the actual sentiment of the Text.

**E.g**: Twenty four seven (*Neutral*) \-> All time (*Positive*)
A piece of cake (*Neutral*) \-> Very Easy task (*Positive*)

#####  Sarcastic Phrases:
The Text might contain Sarcastic Phrases, which creates ambiguity in identifying the actual sentiment of the Text.

*E.g :* The Service Could have been better and faster

The above text contains positive words like “better”, ”faster” but it has been told in sarcastic way. The actual sentiment of this sarcastic text is negative.

*Resolution:* The customized Dictionary has been created for sarcastic phrases <“could have been “,”could be”,…> to identify the sarcastic phrases and replaces sarcastic phrases as follows to get the actual sentiment.

**Multiple Adjective <Add “not” as prefix to Adjective>:**

The Service Could have been better and faster (Positive)  \-> The Service not better and faster (Negative)  

**Single Adjective < Adjective Replaced with Antonym of Adjective>:**

The Service Could have been better (Positive) \-> The Service worsen (Negative)

  1.  **Comparison Keywords :**   

The Text might contain Comparison keywords like *<instead,against,than>*, which carries mixed sentiment. 

E.g: “Want to speak to real person **than** machine”

*Resolution:* The above sentence needs to be split into multiple sentences based on the comparison preposition and append ‘not’ to the next sentence to get the actual sentiment.

Want to speak to real person **than** machine (Positive) **\->**  Want to speak to real person (Positive)

**"Not"** Want to speak to machine (Negative).
####  New Algorithm for correction of Sentiment Analysis :

  **Case 0** - Both TextBlob and Vader are having same non-neutral sentiment **=>** Final Sentiment **=** *Vader Sentiment*

  **Case 1** - Only TextBlob Sentiment (TextBlob **=>**  Positive/Negative and Vader **=>**  Neutral) **=>**  Final Sentiment **=** *TextBlob Sentiment*

**Case 2** - Only Vader Sentiment (Vader **=>** Positive/Negative and TextBlob **=>** Neutral) **=>** Final Sentiment **=** *Vader Sentiment*

**Case 3** - Both TextBlob and Vader Sentiment are Neutral(below logic is considered) **=>** Final Sentiment **=** 				
 - [ ] Priority 1(Vader Sentiment of modified Text)		
 - [ ] Priority 2(TextBlob Sentiment of modified Text)

- a. Create Negative Words Dictionary from the words not handled by both TextBlob and Vader.

- b.  Read the Phrase and check whether individual words are present in Negative Dictionary created in above step.

- c.  If found in Negative Dictionary, add “BAD” keyword just before the word. For Example “pending” ,it will be translated to “BAD pending”.

- d. The Logic for adding “BAD” before the keywords to translate the text and get the sentiment which TextBlob and Vader can handle.

- e.  Re-do the Vader and TextBlob Sentiment ,give more priority to Vader and If Vader sentiment is neutral, then the TextBlob Sentiment is chosen.

**Case 4** - Both TextBlob and Vader Sentiment differs but not Neutral (same logic of *Case3*) **=>** Final Sentiment **=** 
 - [ ] priority 1. (*Vader Sentiment* of modified Text) 
 - [ ]  priority 2. (*TextBlob Sentiment* of modified Text)
