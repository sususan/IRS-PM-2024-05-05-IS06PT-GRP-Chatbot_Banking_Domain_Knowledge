

# from coursedbclient import CourseDBClient

from configurations import logger
# from configurations import COURSEDBCLIENT_ALLOWED_LISTABLE_FIELDS

from tabulate import tabulate

import pandas as pd


class BotUtils:
    '''
    BotUtils class to group utility methods for bot.
    '''
    # initial logging
    loghead = "BotUtils"

    # @classmethod
    # def get_all_urls(cls):
    #     ''' 
    #     returns list of complete str urls
    #     '''
    #     logsubhead = f"{cls.loghead}.get_all_urls()-"
    #     # initial DB client note: never instantiate this on module level
    #     # because it will not go out of scope.
    #     # course_dbclient = CourseDBClient()
    #     # urllist = course_dbclient.get_all_urls()  # get all urls
    #     logger.debug(f"{logsubhead} len(urllist)={len(urllist)}")
    #     return urllist

    # @classmethod
    # def get_all_of_fields(cls, fields: dict):
    #     '''
    #     get_all_of_fields whichever combination of titles, labels, category etc
    #     that are listed in COURSEDBCLIENT_ALLOWED_LISTABLE_FIELDS
    #     whereby fields is dict {fieldname1: bool, fieldname2: bool....}
    #     if bool is True, include it, else exclude it
    #     _ids are excluded.
    #     returns listofurldict is a list of {'title': title<str>, 'label': label<str>, 'category': category<str>}
    #     where fields are available.       logsubhead = f"{cls.loghead}.get_all_of_fields(cls,fields:dict)-"
    #     '''
    #     logsubhead = f"{cls.loghead}.get_all_of_fields(cls,fields)-"
    #     # return empty list if no fields selected
    #     num_fields = len(fields.keys())
    #     if (num_fields < 1):
    #         logger.info(f'{logsubhead} no fields')
    #         return []

    #     # remove all unallowed keys not allowed
    #     logger.info(
    #         f"{logsubhead} allowed fields={COURSEDBCLIENT_ALLOWED_LISTABLE_FIELDS}")
    #     unallowed_fields = []
    #     for k in fields.keys():
    #         if k not in COURSEDBCLIENT_ALLOWED_LISTABLE_FIELDS:
    #             logger.info(f"{logsubhead} field {k} unallowed.")
    #             unallowed_fields.append(k)

    #     for f in unallowed_fields:
    #         logger.info(f"{logsubhead} pop unallowed field {f}")
    #         fields.pop(f)

    #     course_dbclient = CourseDBClient()
    #     return course_dbclient.get_all_of_fields(fields)

    @classmethod
    def parse(cls, s: str, *args):
        '''
        parse s where *args is a list of delimiters of type str
        replace all the delimiters with ' '
        and split the s:str to return as list
        list returned is not None
        '''
        logsubhead = f"{cls.loghead}.parse(cls,str,*delimiters)-"
        logger.debug(f"{logsubhead} parsing ({(s, args)})")
        if s is None or s == "":
            return []
        for d in args:
            s = s.replace(d, ' ')  # replace each delimiter with ' '
        s = s.split()  # return split string with default whitespace
        logger.debug(f"{logsubhead} parsed ({s})")
        return s

    @classmethod
    def tabulate(cls, df: pd.DataFrame):
        '''
        formats a pandas.DataFrame into tabulated form of type str
        returns a table in str
        '''
        return tabulate(df, tablefmt="github", headers="keys", showindex=True)
