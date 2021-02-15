import neo4j
from toml_parser import Parser


class Interface():

    def __init__(self):

        self._config = Parser("config.toml").load()

        uri = self._config["database"]["neo4j"]["uri"]
        user = self._config["database"]["neo4j"]["user"]
        password = self._config["database"]["neo4j"]["password"]

        self._driver = neo4j.Driver(uri, auth=(user, password))


    def close(self):

        self._driver.close()


    def get_customer_product_pairs(self):

        query = (
            f"MATCH (p:Product)<-[:OF]-()-[:GIVEN_BY]->(c:Customer) "
            f"RETURN c.id, p.id"
        )

        with self._driver.session() as session:
            for record in session.run(query):
                yield tuple(record)


    def create_recommendation_relationship(self, customer_id, product_id, recommendation_coefficient):

        query = (
            f"MATCH (c:Customer {{id: {repr(customer_id)}}}), "
            f"(p:Product {{id: {repr(product_id)}}}) "
            f"CREATE (c)<-[:RECOMMENDED {{recommendation_coefficient: {recommendation_coefficient}}}]-(p)"
        )

        with self._driver.session() as session:
            session.run(query)

