
import base64
from html import unescape
import os
import joblib

# From Streamlit
import streamlit as st
import streamlit_option_menu
from streamlit_option_menu import option_menu
from streamlit_card import card
from PIL import Image


from sklearn.feature_extraction.text import TfidfVectorizer

# Load Data Dependencies
import pandas as pd
import numpy as np

# Load the external stylesheet/css
with open('./styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
my_vectoriser = TfidfVectorizer
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file
classifier = ['Logistic Regression', 'Random Forest','K-Nearest Neighbour', 'LinearSVC']
# Load your raw data
raw = pd.read_csv("resources/train.csv")

text = """
    <h1>Unveiling Trends, One Tweet at a Time!</h1>

    
"""
# For contact us
contactUs = """
    <h4>High Office</h4>
    <p>1171 Southwest Midway Forrest</p>
    <p>Goat Land, TX-7890</p>
    
    <p>011-022-4444</p>
    <p>info@tweetscope.io</p>

"""
# EDA 
hasht = ['./resources/imgs/hash_neutral.png','./resources/imgs/hash_anti.png', 'resources/imgs/hash_news.png', 'resources/imgs/hash_pro.png']
buzz = ['resources/imgs/anti_climate.png','resources/imgs/pro_climate.png','resources/imgs/news_climate.png','resources/imgs/neutral_climate.png']
dc = ['resources/imgs/dist_sent_class.png', 'resources/imgs/pie_tweet.png']

# For about us
aboutUs = """
    <p>TweetScope v1.0.0</p>
    <p class="ts" >© 2024 TweetScope LLC</p>
    <h5>Made with ❤️ by the team @ Nexus Analytics </h5>
    <p></p>
"""
listI = ['./resources/imgs/z_.webp', './resources/imgs/hl_.webp','./resources/imgs/s_.webp', './resources/imgs/neo_.webp',
         './resources/imgs/nl_.webp', './resources/imgs/v_.webp'
         ]






# with open(filepath, "rb") as f:
#     data = f.read()
#     encoded = base64.b64encode(data)
# data = "data:image/png;base64," + encoded.decode("utf-8")

# F
def main():
        """Tweet Classifier App with Streamlit """

        with st.sidebar:
            selected = option_menu(
            menu_title = "The Menu",
            options = ["Home","Classifiers","About the Data","EDA","About Us","Contact Us"],
            icons = ["house","database", "gear","binoculars","info-circle","telephone"],
            menu_icon = "app-indicator",
            styles={
                   'nav-link':{'--hover-color': "#FF6700"},
                   'nav-link-selected': {'background-color':'navy'},
                   'a':{'color': '#fff'}

            },
            default_index = 0,
            #orientation = "horizontal",
        )
            st.markdown(
    """
    <div style='background-color:#f0f0f0; padding:10px; border-radius:5px;'>
        <h2 style='text-align:center;'>🎈 Interactive charts coming in v1.1.0!</h2>
    </div>
    """,
    unsafe_allow_html=True
)
        if selected == "Home":
            st.image('./resources/imgs/Twitter_Scope-removebg-preview.png')
            st.write(unescape(text), unsafe_allow_html=True)
               
        if selected == "EDA":

            st.header('From raw data to insights, EDA paves the way.')
            st.markdown('##')
            st.markdown('##')

            st.info('The Quick Scan 👀')
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Tweets ", value="15819 ")
            col2.metric("Duplicated Tweets", "10.05 %")
            col3.metric('Null Values', '0 %')
            st.markdown('##')
            st.markdown('##')
            
            
            st.info('Tweet roulette and class roulette: where every distribution is a spin of the wheel!')
            st.image(dc, caption=['Class Distribution', 'Tweet Distribution'])
            st.markdown("""
                    ### Observations -

- News (Class 2) (23%): This class has 3640 tweets, as per the count you provided. These tweets likely link to factual news about climate change.

- Pro (Class 1) (54%): This class has 8530 tweets, making it the most prevalent sentiment in your dataset. These tweets likely express support for the belief in man-made climate change.

- Neutral (Class 0) (15%): This class has 2353 tweets. Tweets in this category neither support nor refute the belief in man-made climate change.

- Anti (Class -1) (8%): This class has 1296 tweets, indicating a smaller number of tweets that express disbelief in man-made climate change.


""")
            st.markdown('##')
            st.markdown('##')
            



            st.info("The Hashtags #️⃣")
            
            col4, col5, col6 = st.columns(3)
            st.markdown("""<style>
                        /* About Us */
.ts{
    color: rgb(129, 127, 127) !important;
}
.st-emotion-cache-1kyxreq {
    display: flex;
    flex-flow: wrap;
    row-gap: 1rem;
    justify-content: center;
    column-gap: 3rem;
}
div[data-testid ="stImage"] > img {
    width: 300px;
    height: 200px;
    border-radius: 10px;
    border: 2px solid #231486;
    box-shadow: rgba(0, 0, 0, 0.25) 0px 4px 10px, rgba(0, 0, 0, 0.12) 0px -12px 30px, rgba(0, 0, 0, 0.12) 0px 4px 6px, rgba(0, 0, 0, 0.17) 0px 12px 13px, rgba(0, 0, 0, 0.09) 0px -3px 5px;
    transition: transform .7s ease;
}
div[data-testid ="stImage"] > img:hover {
    -ms-transform: scale(1.5); /* IE 9 */
    -webkit-transform: scale(1.5); /* Safari 3-8 */
    transform: scale(1.5); 
}
                        </style>""", unsafe_allow_html=True)
            
            st.image(hasht, caption=['Neutral #️⃣', 'Anti #️⃣', 'News #️⃣', 'Pro #️⃣'])
            st.markdown("""### Analysis:

- The hashtags "MAGA" and "Trump" rank as the first and third most frequently used hashtags in anti-climate change tweets, respectively. "MAGA," which stands for "Make America Great Again," originated as Donald Trump's campaign slogan during the 2016 elections. On Twitter, it became a rallying cry for his supporters, symbolized by the hashtag "#MAGA" and associated with support for Trump. This suggests that a significant portion of anti-climate change tweets stem from individuals who align with Trump's ideology.

- Additionally, "DrainTheSwamp" appears among the top anti-climate change hashtags. Trump popularized this phrase as a metaphor for his plan to address corruption within the federal government. During his presidential campaign, he vowed to address issues such as raising taxes on the wealthy, including hedge fund managers. The presence of this hashtag further reinforces the notion that many Trump supporters on Twitter also express anti-climate change sentiments.

- Further down the list, "TCOT" (Top Conservative On Twitter) holds the sixth position. This term serves as a means for conservatives, particularly Republicans, to connect with like-minded individuals on the platform. This trend underscores a recurring pattern of themes associated with Trump's presidency and conservative viewpoints.

- Finally, hashtags like "FakeNews" and "ClimateScam" also garner significant attention. These hashtags may be employed by individuals who oppose the concept of climate change to discredit information and sources they perceive as biased or inaccurate.""")
            st.markdown('##')
            st.markdown('##')
            
            st.info("Something is Buzzzzzing ... is it a 🐝🐝🐝 ?   Nope its a Word")
            st.image(buzz, caption=['Anti', 'Pro', 'News', 'Neutral'])
            st.markdown("""
                        ### Observations:

- The top 3 buzzwords accross all classes are climate change and rt (retweet). The frequency of rt ( Retweet ) means that a lot of the same information and/or opinions are being shared and viewed by large audiences. This is true for all 4 classes

- 'Trump' is a frequently occuring word in all 4 classes. This is unsurprising given his controversial view on the topic.

- Words like real, believe, think, fight, etc. occur frequently in pro climate change tweets. In contrast, anti climate change tweets contain words such as 'hoax', 'scam', 'tax', 'liberal'and 'fake'. There is a stark difference in tone and use of emotive language in these 2 sets of tweets. From this data we could reason that people who are anti climate change believe that global warming is a 'hoax' and feel negatively towards a tax–based approach to slowing global climate change

- words like 'science' and 'scientist' occur frequently as well which could imply that people are tweeting about scientific studies that support their views on climate change.

- EPA, the United States Environmental Protection Agency is another climate change 'buzzword' that appears frequently across classes.
                        """)


                


                 




            




                

        # Building out the predication page
        if selected == "About the Data":

            
            st.info("General Information")
            # You can read a markdown file from supporting resources folder
            st.markdown("""
                ### About the raw data

                The train data contain more than 10,000 tweets... That's a lot of words!

                The tweets are divided into 4 classes:

                 **[ 2 ] News:** Tweets linked to factual news about climate change.
                        
                 **[ 1 ] Pro:** Tweets that support the belief of man-made climate change.
                        
                 **[ 0 ] Neutral:** Tweets that neither support nor refuse beliefs of climate change.
                        
                 **[ -1 ] Anti:** Tweets that do not support the belief of man-made climate change.

                Retweets account for 10% of the train data.
                """)


            st.subheader("Raw Twitter data and label")
            if st.checkbox('Show raw data'): # data is hidden if box is unchecked
                st.write(raw[['sentiment', 'message']]) # will write the df to the page

        # Building out the predication page
        if selected == "Classifiers":
            st.info("Prediction with ML Models")
            tab1, tab2, tab3, tab4 = st.tabs(['Logistic Regression','Random Forest','K-Nearest Neighbour', 'LinearSVC',])
            
            

            with tab1:
                st.header("Logistic Regression")
                # Creating a text box for user input
                tweet_text4 = st.text_area("Enter Text","Type Here")
                l_texet = pd.Series(tweet_text4)
                

                print(l_texet)
                if st.button("Classify"):
                    # Transforming user input with vectorizer
                    
                    # Load your .pkl file with the model of your choice + make predictions
                    # Try loading in multiple models to give the user a choice
                    predictor = joblib.load(open(os.path.join("resources/lr_model.pkl"),"rb"))
                    prediction = predictor.predict(l_texet)

                    # When model has successfully run, will print prediction
                    # You can use a dictionary or similar structure to make this output
                    # more human interpretable.
                    st.success("Text Categorized as: {}".format(prediction))


            with tab2:
                st.header("Random Forest")  

                

                # Creating a text box for user input
                tweet_text4 = st.text_area("Enter Text","Type Here", key=1)
                l_texet = pd.Series(tweet_text4)
                

                print(l_texet)
                if st.button("Classify", key=2):
                    # Transforming user input with vectorizer
                    
                    # Load your .pkl file with the model of your choice + make predictions
                    # Try loading in multiple models to give the user a choice
                    predictor = joblib.load(open(os.path.join("resources/rf_model.pkl"),"rb"))
                    prediction = predictor.predict(l_texet)

                    # When model has successfully run, will print prediction
                    # You can use a dictionary or similar structure to make this output
                    # more human interpretable.
                    st.success("Text Categorized as: {}".format(prediction))  

            with tab3:
                st.header("K-Nearest Nabour")

                # Creating a text box for user input
                tweet_text4 = st.text_area("Enter Text","Type Here", key=3)
                l_texet = pd.Series(tweet_text4)
                

                print(l_texet)
                if st.button("Classify", key=4):
                    # Transforming user input with vectorizer
                    
                    # Load your .pkl file with the model of your choice + make predictions
                    # Try loading in multiple models to give the user a choice
                    predictor = joblib.load(open(os.path.join("resources/knn_model.pkl"),"rb"))
                    prediction = predictor.predict(l_texet)

                    # When model has successfully run, will print prediction
                    # You can use a dictionary or similar structure to make this output
                    # more human interpretable.
                    st.success("Text Categorized as: {}".format(prediction))
                

            with tab4:
                st.header('LinearSVC')

                # Creating a text box for user input
                tweet_text4 = st.text_area("Enter Text","Type Here", key=5)
                l_texet = pd.Series(tweet_text4)
                

                print(l_texet)
                if st.button("Classify", key=6):
                    # Transforming user input with vectorizer
                    
                    # Load your .pkl file with the model of your choice + make predictions
                    # Try loading in multiple models to give the user a choice
                    predictor = joblib.load(open(os.path.join("resources/lsvc_op_model.pkl"),"rb"))
                    prediction = predictor.predict(l_texet)

                    # When model has successfully run, will print prediction
                    # You can use a dictionary or similar structure to make this output
                    # more human interpretable.
                    st.success("Text Categorized as: {}".format(prediction))
                

        if selected == "Contact Us":
             
             with st.container():
                  st.write(unescape(contactUs), unsafe_allow_html=True)
        
        if selected == "About Us":
             st.markdown("""<style>
                        /* About Us */
.ts{
    color: rgb(129, 127, 127) !important;
}
.st-emotion-cache-1kyxreq {
    display: flex;
    flex-flow: wrap;
    row-gap: 1rem;
    justify-content: center;
    column-gap: 3rem;
}
div[data-testid ="stImage"] > img {
    width: 200px;
    height: 220px;
    border-radius: 10px;
    border: 2px solid #231486;
    box-shadow: rgba(0, 0, 0, 0.25) 0px 14px 15px, rgba(0, 0, 0, 0.12) 0px -12px 30px, rgba(0, 0, 0, 0.12) 0px 4px 6px, rgba(0, 0, 0, 0.17) 0px 12px 13px, rgba(0, 0, 0, 0.09) 0px -3px 5px;
    transition: transform .7s ease;
}
div[data-testid ="stImage"] > img:hover {
    -ms-transform: scale(1.5); /* IE 9 */
    -webkit-transform: scale(1.5); /* Safari 3-8 */
    transform: scale(1.5); 
}
                        </style>""", unsafe_allow_html=True)
             st.write(unescape(aboutUs), unsafe_allow_html=True)
             with st.container():
                  st.image(listI,width=200, caption=["Zithulele", 'Hlakaniphile', 'Siyamthanda', 'Neo', 'Nolwazi', 'Vuyo'], )
                  
                  
                  
                  
                #   st.write(unescape(aboutUs), unsafe_allow_html=True)
                    

            



# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()