import enum
import logging

### LOGGER ###
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger('mike_nusiss')
logger.setLevel(logging.INFO)

### General Project Path###
projectpath = '.'

### read mode for testing from file or http request from udemy.###


class ReadMode(enum.Enum):
    fileio = 1
    httprequest = 2


readmode = ReadMode.fileio  # default read from downloaded jsonfile

### for retrieval of records ###
UDEMY_SERVER_STEM = "https://www.udemy.com"
UDEMY_SERVER_ALL_COURSES_URL = 'https://www.udemy.com/api-2.0/discovery-units/all_courses/'
downloaded_course_json_file_stem = 'udemy'
udemyquery_payload = {
    'p': 1,
    'page_size': 16,
    'lang': 'en',
    'price': 'price-free',
    'sort': 'popularity',
    'category_id': '283',
    'source_page': 'category_page',
    'locale': 'en_US',
    'currency': 'sgd',
    'navigation_locale': 'en_US',
    'skip_price': 'true',
    'sos': 'pc',
    'fl': 'cat'
}

### DATABASE SPECIFIC ###
DATABASE_URL = 'mongodb://localhost:27017'
DATABASE_NAME = 'mike'
DATABASE_COLLECTION = 'udemycourses'
COURSEDBCLIENT_ALLOWED_LISTABLE_FIELDS = ['title',
                                          'num_subscribers',
                                          'rating',
                                          'num_reviews',
                                          'num_published_lectures',
                                          'published_time',
                                          'last_update_date',
                                          'content_duration_min',
                                          'label',
                                          'category']


# turn restricted access on and off
RESTRICTED_ACCESS = False

### BOT AUTHORISED users ###
### add usernames here. ###
BOT_AUTHORISED_USERS = [
    'Michael_Lee_SG',
    'mikelee_sg', 
    'LWCjason'
]

# Practice Module
# neo4j connection parameters
neo4j_username = 'neo4j'
neo4j_password = 'LwDgVxLXxvHpXDp7IBXT62o7ofNRIRpNgUs3FhEfIjo'
neo4j_connection_string = 'neo4j+s://006136ba.databases.neo4j.io:7687'