Feature: Client API
    Scenario: Data Source Columns (SQL Provider)
        Given Featureform is installed
        When I create a "hosted" "insecure" client for "localhost:7878"
        When I register postgres
        And I generate a random variant name
        And I register a table from postgres
        Then I should get the columns for the data source from "postgres"

    Scenario: Data Source Columns (Spark)
        Given Featureform is installed
        When I create a "hosted" "insecure" client for "localhost:7878"
        And I generate a random variant name
        And I register "s3" filestore with bucket "featureform-spark-testing" and root path "data"
        And I register databricks
        And I register transactions_short.csv
        Then I should get the columns for the data source from "spark"