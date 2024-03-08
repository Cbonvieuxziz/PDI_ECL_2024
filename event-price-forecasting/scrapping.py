import newspaper
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class NewsScrapper:

    def __init__(self):

        self.urls = [
            'https://edition.cnn.com/',
            'https://time.com/',
            'https://www.cnbc.com/world/?region=world',
            'http://www.huffingtonpost.com',
            'https://www.foxnews.com/',
            'http://theatlantic.com',
            'https://www.pcmag.com/',
            'http://www.bbc.co.uk',
            'https://www.businessinsider.com/?r=US&IR=T',
            'https://newrepublic.com/',
            'https://thebusinessjournal.com/'            
        ]        

        self.relevant_keywords = []
        self.papers = []
        self.relevant_articles = []

        try:
            self.analyzer = SentimentIntensityAnalyzer()
        
        except:
            nltk.download('vader_lexicon')
            nltk.download('punkt')
            self.analyzer = SentimentIntensityAnalyzer()


    def build_papers(self) -> None:
        '''
        Call newspaper.build on every known url and store the resulting paper
        '''

        urls_count = len(self.urls)

        print("Building papers :")

        for index, url in enumerate(self.urls):            
            try:
                print(f"[{index+1}/{urls_count}] - {url}")
                paper = newspaper.build(url, language='en')
                self.papers.append(paper)
            
            except:
                print(f"Could not get paper from url : {url}")
    

    def is_article_relevant(self, article : newspaper.article.Article) -> bool:
        '''
        Tell whether the content of the article is relevant to the problem or not       
        '''
        keywords = article.keywords

        # Raw material price
        for raw_material in ["gaz", "oil", "coal"]:
            if raw_material in keywords:
                return True
                
        
        # Factors that may influence the supply and demand balance
        # Weather : 
        for weather_condition in ["heatwave", "coldwave", "coldsnap"]:
            if weather_condition in keywords:
                return True
        
        # Shutdown of a production plant
        for event in ["shutdown", "power", "plant", "station", "powerhouse"]:
            if event in keywords:
                return True

        if "CO2" in keywords and "price" in keywords:
            return True
        
        return False
  
            
    def retrieve_relevant_articles(self) -> None:
        '''
        Iterate over the list of papers. For each article in the paper, store it if its content might influence the electricity price
        '''
        
        for paper in self.papers:
            print("paper:", paper)
            for article in paper.articles:
                try:
                    article.download()
                    article.parse()
                    article.nlp()
                    print("article:", article.title)
                    if self.is_article_relevant(article):
                        self.relevant_articles.append( article ) 
                except Exception as e:       
                    print(e)


    def get_day_score(self) -> float:
        '''
        Compute an overall score of the day. 
        This score will be a parameter of the model

        Returns: A score between -1 and 1
        '''    
        pass

                    
