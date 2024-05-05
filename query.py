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
            "MATCH (f:FAQ) WHERE f.Question CONTAINS('" + question + "') RETURN f.Answer AS Answer "
    )
    with driver.session() as session:
        results = session.run(query)
        for result in results:
            print('Answer' + ': ' + result['Answer'])
            return result['Answer']
    return "No relevant answer found!"


def getanswerbylable(label, driver):
    query = (
            "MATCH (f:FAQ) WHERE f.Label CONTAINS('" + label.replace("\'", "\\'") + "') RETURN f.Answer AS Answer, f.Question As Question "
    )
    with driver.session() as session:
        results = session.run(query)
        for result in results:
            print('Answer' + ': ' + result['Answer'])
            return result['Question'], result['Answer']


def findSubSection(intent, driver):
    query = (
            "MATCH p=(f:FAQ)-[:HAS_SUBSECTION]->(ss:SubSection) WHERE f.Label='" + intent + "' RETURN ss.Label AS SubSection"
    )
    with driver.session() as session:
        results = session.run(query)
        for result in results:
            return result['SubSection']


def findSimilerQuestion(lable, initialQuestion, driver):
    query = (
            "MATCH p=(f:FAQ)-[:HAS_SUBSECTION]->(ss:SubSection) WHERE ss.Label='" + lable + "' AND f.Question <> '"
            + initialQuestion.replace("\'", "\\'") + "' RETURN f.Question AS Question LIMIT 3"
    )
    ret = []
    with driver.session() as session:
        results = session.run(query)
        for result in results:
            if result['Question'] != initialQuestion:
                ret.append(result['Question'])
    return ret


def main():
    driver = connect_to_neo4j()
    getanswerbyquestion("question", driver)


if __name__ == '__main__':
    main()
