from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import json

# Neo4j connection details
uri = "neo4j+s://c22b0b3c.databases.neo4j.io:7687"
username = "neo4j"
password = "akterzObCsJTwulDEG7AFlCkDcMyGC7RblEnmXbg7aE"


# Initialize the graph DB and delete all the nodes and relationships

def connect_to_neo4j():
    return GraphDatabase.driver(uri, auth=(username, password))


def getanswerbyquestion(question, driver):
    print(question)
    query = (
        "MATCH (f {Question:'Am I eligible for a credit card?'}) RETURN f.Answer AS Answer "
    )
    with driver.session() as session:
        results = session.run(query)
        for result in results:
            print('Answer' + ': ' + result['Answer'])
            return result['Answer']
    return "No relevant answer found!"


def main():
    driver = connect_to_neo4j()
    getanswerbyquestion("question", driver)


if __name__ == '__main__':
    main()
