import mysql.connector
import pandas as pd
import os

# Create the output directory if it doesn't exist
output_dir = "omop_tables_purdue"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# tables to export
tables_to_export = [
    "person",
    "observation_period",
    "visit_occurrence",
    "visit_detail",
    "condition_occurrence",
    "drug_exposure",
    "procedure_occurrence",
    "device_exposure",
    "measurement",
    "observation",
    "death",
    "note",
    "note_nlp",
    "specimen",
    "fact_relationship",
    "location",
    "care_site",
    "provider",
    "payer_plan_period",
    "cost",
    "drug_era",
    "dose_era",
    "condition_era",
    "episode",
    "episode_event",
    "metadata",
    "cdm_source",
    "concept",
    "vocabulary",
    "domain",
    "concept_class",
    "concept_relationship",
    "relationship",
    "concept_synonym",
    "concept_ancestor",
    "source_to_concept_map",
    "drug_strength",
    "cohort",
    "cohort_definition"
]

mode = input("Select export mode ('all' or 'concept'): ").strip().lower()
if mode == 'concept':
    print("Export mode: concept. Only exporting the trimmed vocabulary tables.")
    domain_filter = "c.domain_id IN ('Person','Measurement','Observation','Condition','Drug')"
    tables_to_export = [
        "concept",
        "concept_synonym",
        "concept_ancestor",
        "concept_relationship",
        "domain",
        "vocabulary",
        "concept_class"
    ]

print("Connecting to database...")
try:
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="omop"
    )
    print("Connected successfully!")

    for table_name in tables_to_export:
        print(f"Exporting {table_name}...")
        try:
            if mode == 'concept':
                if table_name == 'concept':
                    query = f"SELECT * FROM concept c WHERE {domain_filter}"
                elif table_name == 'concept_synonym':
                    query = (
                        "SELECT s.* FROM concept_synonym s "
                        "JOIN concept c ON s.concept_id=c.concept_id "
                        f"WHERE {domain_filter}"
                    )
                elif table_name == 'concept_ancestor':
                    query = (
                        "SELECT * FROM concept_ancestor ca "
                        "WHERE ca.ancestor_concept_id IN "
                        f"(SELECT concept_id FROM concept WHERE {domain_filter}) "
                        "   OR ca.descendant_concept_id IN "
                        f"(SELECT concept_id FROM concept WHERE {domain_filter})"
                    )
                elif table_name == 'concept_relationship':
                    query = (
                        "SELECT * FROM concept_relationship cr "
                        "WHERE cr.concept_id_1 IN "
                        f"(SELECT concept_id FROM concept WHERE {domain_filter}) "
                        "   OR cr.concept_id_2 IN "
                        f"(SELECT concept_id FROM concept WHERE {domain_filter})"
                    )
                else:
                    query = f"SELECT * FROM {table_name}"
            else:
                query = f"SELECT * FROM {table_name}"

            df = pd.read_sql(query, connection)
            output_file = os.path.join(output_dir, f"{table_name}.csv")
            df.to_csv(output_file, index=False)
            print(f"  Exported {len(df)} rows to {output_file}")
        except Exception as e:
            print(f"  Error exporting {table_name}: {e}")

    print("Export complete!")
    abs_path = os.path.abspath(output_dir)
    print("All CSV files were saved to:")
    print(abs_path)

except mysql.connector.Error as err:
    print(f"Error connecting to database: {err}")

finally:
    if 'connection' in locals() and connection.is_connected():
        connection.close()
        print("Database connection closed")
