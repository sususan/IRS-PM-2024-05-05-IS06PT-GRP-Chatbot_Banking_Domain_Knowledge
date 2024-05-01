#####
# bot using CommandHandler
# To start, create a file "credentials.py" in the same directory and add following line.
# Replace <TOKEN> with your own token.
#   telegram_bot_token = '<TOKEN>'
####


from functools import wraps
from configurations import logger
from configurations import BOT_AUTHORISED_USERS, RESTRICTED_ACCESS
from telegram import ReplyKeyboardMarkup, Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    PicklePersistence,
    CallbackContext,
)
from credentials import telegram_bot_token
# from visualization import Visualization
from botutils import BotUtils
from tabulate import tabulate

import pandas as pd
import numpy as np

# log header
loghead = "bot"


# standard help message
BOT_MESSAGE_INTRO = '''
Hi there, welcome to bank bot, how may we help you with your queries?

Since this is in alpha testing stage and only developed for
project showcase, we have limited some functionality to 
a select group of users.

You may find that your username cannot access some of the functions.
'''

BOT_MESSAGE_HELP = '''
Hi! I can show you plenty of free udemy courses.
 What would you like to do next?
 /start
 introduction

 You may type any query to start


 /help - print this help message.
'''


def restricted(func):
    @wraps(func)
    def wrapped(update: Update, context: CallbackContext, *args, **kwargs):
        '''
        decorator restricts the access to only a few selected groups
        ie. BOT_AUTHORISED_USERS in configurations
        '''
        logsubhead = f"{loghead}.restricted decorative wrapper-"
        logger.debug(f"{logsubhead} start")
        username = update.effective_user.username
        logger.info(f"{logsubhead} username={username}")
        if RESTRICTED_ACCESS and username not in BOT_AUTHORISED_USERS:
            logger.info(
                f"{logsubhead} Unauthorized access denied for {username}.")
            s = update.effective_user.first_name
            update.message.reply_text(
                f"I'm sorry {s}, I'm afraid I can't do that; you are not authorised.")
            return
        return func(update, context, *args, **kwargs)
    return wrapped


def handle_start_cmd(update: Update, context: CallbackContext):
    '''
    Start command 
    '''
    logsubhead = f"{loghead}.handle_start_cmd(update, context)-"
    update.message.reply_text(f"{BOT_MESSAGE_INTRO}\n{BOT_MESSAGE_HELP}")
    logger.debug(f"{logsubhead} update.message = {type(update.message)}")
    logger.debug(f"{logsubhead} Update update = {update}")
    logger.debug(f"{logsubhead} CallbackContext context = {context}")


# def handle_viz_cmd(update: Update, context: CallbackContext):
#     '''
#     visualize some information from the data
#     '''
#     logsubhead = f"{loghead}.handle_viz_cmd(update, context)-"
#     update.message.reply_text(f"{BOT_MESSAGE_INTRO}\n{BOT_MESSAGE_HELP}")
#     logger.debug(f"{logsubhead}update = {update}")
#     logger.debug(f"{logsubhead}context = {context}")

#     # get random viz image and send as photo
#     # and get its length of bytes
#     bio = Visualization.get_random_dataviz()
#     bytelength = len(bio.getvalue())

#     logger.info(f"{logsubhead}len(bio.getvalue()) bytelength= {bytelength}")

#     # more than 256kb, send as a document, else send as a photo
#     if bytelength > 256000:
#         update.message.reply_document(bio)
#     else:
#         update.message.reply_photo(photo=bio)


def handle_list_cmd(update: Update, context: CallbackContext):
    '''
    # to handle all the list commands
    # only title, label, category are of meaning.
    # so let's limit to that
    # if no parameters, list all the titles
    '''
    logsubhead = f"{loghead}.handle_list_cmd(update, context)-"

    # full_command is something like
    # '/list title,label' or
    # '/list title,label,category or
    # '/list title'
    full_command = update.message.text
    logger.info(f"{logsubhead} parsing command {full_command}")

    fields_list = BotUtils.parse(full_command, ',')

    # get dict of {field:True} where field is of 'title', 'label'...etc
    # remove the key '/list'
    # if only '/list' at least include title where {'title':True}
    fields_dict = {}
    if len(fields_list) > 1:
        for f in fields_list:
            if f != '/list':
                fields_dict[f] = True
    if len(fields_dict.keys()) < 1:
        fields_dict['title'] = True

    logger.info(f"{logsubhead} fields_dict={fields_dict}")

    # get_all_of_fields whichever combination of titles, labels, category
    # whereby fields is dict {fieldname1: bool, fieldname2: bool....}
    # if bool is True, include it, else exclude it
    # _ids are excluded.
    # returns listofurldict is a list of {'title': title<str>, 'label': label<str>, 'category': category<str>}
    # where fields are available.
    answer_list = BotUtils.get_all_of_fields(fields_dict)
    logger.info(f"{logsubhead} query result len ={len(answer_list)}")
    logger.debug(f"{logsubhead} query result answer_list ={answer_list}")

    # we want to message all the fields' values instead of dict
    # using DataFrame, we convert the
    # [
    #   {field1:value1, field2:value2,...}
    #   {field1:value1, field2:value2,...}
    #   {field1:value1, field2:value2,...}
    # ]
    # into a DataFrame that we more easily query and format

    answer_df = pd.DataFrame(answer_list)
    logger.debug(f"{logsubhead} query result as DataFrame ={answer_df}")

    # step 10 rows along the DataFrame
    # take the view, format into tabulated view for display
    num_rows = len(answer_list)
    for a in np.arange(0, num_rows, 10):
        subset_answer_df = answer_df[a: a + 9]
        subset_answer_table = BotUtils.tabulate(subset_answer_df)
        logger.info(f"{logsubhead} subset_answer_table ={subset_answer_table}")
        update.message.reply_text(subset_answer_table)


def handle_help_cmd(update, context):
    '''
    Print help message
    '''
    update.message.reply_text(BOT_MESSAGE_HELP)


@ restricted
def handle_all_cmd(update: Update, context: CallbackContext):
    '''
    show all udemy course urls
    '''
    logsubhead = f"{loghead}.handle_all_cmd(update, context) -"
    logger.info(f"{logsubhead} getting all urls")
    urllist = BotUtils.get_all_urls()
    for url in urllist:
        update.message.reply_text(url)


def handle_any_msg(update, context):
    '''
    handles any other message 
    call the model to trigger.
    '''
    update.message.reply_text(BOT_MESSAGE_HELP)


def main():
    '''
    main () of telegram bot
    '''
    logsubhead = f"{loghead}.main()-"
    logger.info(f"{logsubhead} start polling")

    # initialize token
    # Replace <TOKEN> with your own token otherwise uses credentials token
    token = telegram_bot_token
    updater = Updater(token, use_context=True)

    # add handlers
    # updater.dispatcher.add_handler(CommandHandler('all', handle_all_cmd))
    # updater.dispatcher.add_handler(CommandHandler("list", handle_list_cmd))
    updater.dispatcher.add_handler(CommandHandler("start", handle_start_cmd))
    # updater.dispatcher.add_handler(CommandHandler("viz", handle_viz_cmd))
    updater.dispatcher.add_handler(CommandHandler('help', handle_help_cmd))
    updater.dispatcher.add_handler(
        MessageHandler(Filters.text, handle_any_msg))

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
