import mysql.connector
import pandas as pd
import os
from sqlalchemy import create_engine
import warnings

# Suppress specific pandas SQLAlchemy warning
warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy connectable')

# =============================================================================
# Step 1: Define the columns we want in our final dataset
# =============================================================================
# List of columns we want in our final output - exactly matching required format
DESIRED_COLUMNS = [
    # Person and survey identifiers
    'id_subj',           # Patient ID
    'id_dsrvyr',         # Survey ID
    'id_mvpsu',          # Primary sampling unit
    'id_mvstra',         # Stratum
    
    # Demographics
    's_female',          # Gender concept ID 
    's_race',            # Race concept ID
    's_access',          # Healthcare access
    's_poverty',         # Poverty status
    's_insurance',       # Insurance status
    's_private',         # Private insurance
    's_medicaid',        # Medicaid
    's_married',         # Marital status
    's_educ',            # Educational status
    's_insgov',          # Government insurance
    's_smoke',           # Smoking status
    's_smoke4',          # Smoking status (4 category)
    's_alcohol',         # Alcohol consumption
    's_sleep',           # Sleep duration
    's_age',             # Age
    
    # Measurements
    'l_bmxwt',           # Body weight
    'l_bmxbmi',          # Body mass index
    'l_lbxtc',           # Cholesterol
    'l_lbxtr',           # Triglycerides
    'l_lbxgh',           # HDL Cholesterol
    'l_bmxwaist',        # Waist circumference
    'l_lbxglu',          # Glucose
    'l_meansbp',         # Systolic blood pressure
    'l_meandbp',         # Diastolic blood pressure
    'l_ldlnew',          # LDL Cholesterol
    'l_bpmed',           # BP medication
    'l_cholmed',         # Cholesterol medication
    'l_dmmed',           # Diabetes medication
    'l_dminsulin',       # Insulin medication
    'l_dmoral',          # Oral diabetes medication
    'l_dmrisk',          # Diabetes risk score
    'l_bu',              # Blood urea nitrogen
    'l_ua',              # Uric acid
    'l_cr',              # Creatinine
    'l_nasi',            # Sodium
    'l_ksi',             # Potassium
    
    # Diet and activity
    'd_totaldietadjs',   # Total diet adjustments
    'a_score',           # Activity score
    
    # Conditions
    'hx_htn',            # Hypertension
    'deprecated_dx_diagyes',    # Deprecated diagnosis fields
    'deprecated_dx_diagnoand',
    'deprecated_dx_diagnoor',
    'deprecated_dx_poordiagno',
    'deprecated_dx_poor',
    'deprecated_dx_prediag',
    'dx_undxdm',         # Undiagnosed diabetes
    'dx_predm',          # Prediabetes
    'dx_metabolic',      # Metabolic syndrome
    
    # Statistical variables
    '_mi_id',            # Multiple imputation ID
    '_mi_miss',          # Missing flag
    '_mi_m',             # Imputation number
    'cluster_v1',        # Cluster variable 1
    'cluster_v2',        # Cluster variable 2
    
    # Count variables
    'cnt_1', 'cnt_2', 'cnt_3', 'cnt_4', 'cnt_5', 
    'cnt_6', 'cnt_7', 'cnt_8', 'cnt_9', 'cnt_10',
    
    # More conditions and activity
    'diabetes_label',    # Diabetes diagnosis
    'gestational_diabetes', # Gestational diabetes
    'phy_score_year',    # Physical activity score per year
    'phy_score',         # Physical activity score
    
    # Family history
    'familydm_meanSBP',  # Family history systolic blood pressure
    'familydm_meanDBP',  # Family history diastolic blood pressure
    'familydm_BPMed',    # Family history BP medication
    'familydm',          # Family history of diabetes
    
    # Raw values
    's_age_raw',         # Year of birth
    's_female_raw',      # Gender concept ID (raw)
    'familydm_raw',      # Family history raw
    'phy_score_raw',     # Physical activity score raw
    'l_bpmed_raw',       # BP medication raw
    'l_bmxbmi_raw',      # BMI raw
    'gestational_diabetes_raw', # Gestational diabetes raw
    
    # ADA risk score components
    'ada_age',           # ADA age risk component
    'ada_gender',        # ADA gender risk component
    'ada_bmi',           # ADA BMI risk component
    'ada_familydm',      # ADA family history risk component
    'ada_physical',      # ADA physical activity risk component
    'ada_bpmed',         # ADA BP medication risk component
    'ada_overall'        # ADA overall risk score
]

# Define search terms for each clinical concept
# Format: 'output_column_name': ['search term 1', 'search term 2', ...]
MEASUREMENT_SEARCH_TERMS = {
    # Body measurements
    'l_bmxwt': ['body weight', 'weight'],
    'l_bmxbmi': ['body mass index', 'bmi'],
    'l_bmxwaist': ['waist circumference', 'waist'],
    
    # Lipid panel
    'l_lbxtc': ['cholesterol', 'total cholesterol'],
    'l_lbxtr': ['triglyceride'],
    'l_lbxgh': ['hdl', 'high density lipoprotein'],
    'l_ldlnew': ['ldl', 'low density lipoprotein'],
    
    # Vital signs
    'l_lbxglu': ['glucose', 'blood glucose'],
    'l_meansbp': ['systolic', 'systolic blood pressure'],
    'l_meandbp': ['diastolic', 'diastolic blood pressure'],
    
    # Chemistry panel
    'l_bu': ['blood urea nitrogen', 'bun'],
    'l_ua': ['uric acid'],
    'l_cr': ['creatinine'],
    'l_nasi': ['sodium', 'na'],
    'l_ksi': ['potassium', 'k']
}

OBSERVATION_SEARCH_TERMS = {
    's_access': ['healthcare access', 'access to care'],
    's_poverty': ['poverty'],
    's_insurance': ['insurance status', 'health insurance'],
    's_private': ['private insurance', 'commercial insurance'],
    's_medicaid': ['medicaid'],
    's_married': ['marital status', 'married'],
    's_educ': ['education', 'educational status'],
    's_smoke': ['smoking', 'tobacco'],
    's_alcohol': ['alcohol', 'drinking'],
    's_sleep': ['sleep'],
    'l_bpmed': ['blood pressure medication', 'antihypertensive'],
    'l_cholmed': ['cholesterol medication', 'statin'],
    'l_dmmed': ['diabetes medication', 'antidiabetic']
}

CONDITION_SEARCH_TERMS = {
    'hx_htn': ['hypertension', 'high blood pressure'],
    'dx_predm': ['prediabetes', 'impaired glucose tolerance'],
    'dx_metabolic': ['metabolic syndrome'],
    'diabetes_label': ['diabetes mellitus', 'type 2 diabetes'],
    'gestational_diabetes': ['gestational diabetes']
}

# =============================================================================
# Step 2: Database Connection
# =============================================================================
def connect_to_database():
    """
    Create and return a database connection to the OMOP database.

    For different database types:
       - MySQL: mysql+mysqlconnector://user:pass@host/dbname
       - PostgreSQL: postgresql://user:pass@host/dbname
       - SQLite: sqlite:///path/to/database.db
       - MS SQL: mssql+pyodbc://user:pass@host/dbname?driver=ODBC+Driver+17+for+SQL+Server
    """
    try:
        # Create SQLAlchemy engine
        engine = create_engine('mysql+mysqlconnector://root:@localhost/omop')
        return engine
    except Exception as err:
        print(f"Error connecting to database: {err}")
        raise

# =============================================================================
# Step 3: Find the Most Frequent Concept IDs for Each Feature
# =============================================================================
def find_concept_id_from_search_terms(connection, domain, search_terms_dict):
    """
    Find the most frequent concept ID for each feature based on search terms.
    
    Args:
        connection: Database connection (SQLAlchemy engine)
        domain: 'Measurement', 'Observation', or 'Condition'
        search_terms_dict: Dictionary mapping output column names to search terms
    
    Returns:
        Dictionary mapping output column names to concept IDs
    """
    # Define the table name based on domain
    if domain == 'Measurement':
        table_name = 'MEASUREMENT'
        concept_id_column = 'measurement_source_concept_id'
    elif domain == 'Observation':
        table_name = 'OBSERVATION'
        concept_id_column = 'observation_source_concept_id'
    elif domain == 'Condition':
        table_name = 'CONDITION_OCCURRENCE'
        concept_id_column = 'condition_source_concept_id'
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    # Create a dictionary to store the results
    results = {}
    concept_names = {}
    
    # First, let's get all available concept_ids in the database for this domain
    available_concepts_query = f"""
    SELECT DISTINCT c.concept_id, c.concept_name
    FROM {table_name} t
    JOIN CONCEPT c ON t.{concept_id_column} = c.concept_id
    WHERE c.concept_id > 0
    """
    
    try:
        conn = connection.connect()
        raw_conn = conn.connection
        available_concepts_df = pd.read_sql(available_concepts_query, raw_conn)
    except Exception as e:
        available_concepts_df = pd.DataFrame(columns=['concept_id', 'concept_name'])
    
    # Process each feature
    for column_name, search_terms in search_terms_dict.items():
        # Create a more precise search to avoid false matches
        matching_concepts = []
        for search_term in search_terms:
            try:
                # For single letter search terms, require them to be a standalone word
                # or with specific boundaries to avoid matching any word containing that letter
                if len(search_term) == 1:
                    # For 'k' specifically (potassium symbol), look for it as a standalone word
                    # or in specific formats like "K+" or "Serum K"
                    if search_term.lower() == 'k':
                        query = f"""
                        SELECT c.concept_id, c.concept_name, COUNT(*) as occurrence_count
                        FROM {table_name} t
                        JOIN CONCEPT c ON t.{concept_id_column} = c.concept_id
                        WHERE (
                            LOWER(c.concept_name) = LOWER('{search_term}') OR
                            LOWER(c.concept_name) LIKE LOWER('% {search_term} %') OR
                            LOWER(c.concept_name) LIKE LOWER('% {search_term}+%') OR
                            LOWER(c.concept_name) LIKE LOWER('%potassium%') OR
                            LOWER(c.concept_name) LIKE LOWER('%serum {search_term}%')
                        )
                        GROUP BY c.concept_id, c.concept_name
                        ORDER BY occurrence_count DESC
                        """
                    else:
                        # For other single letters, be more strict
                        query = f"""
                        SELECT c.concept_id, c.concept_name, COUNT(*) as occurrence_count
                        FROM {table_name} t
                        JOIN CONCEPT c ON t.{concept_id_column} = c.concept_id
                        WHERE (
                            LOWER(c.concept_name) = LOWER('{search_term}') OR
                            LOWER(c.concept_name) LIKE LOWER('% {search_term} %') OR
                            LOWER(c.concept_name) LIKE LOWER('% {search_term},%')
                        )
                        GROUP BY c.concept_id, c.concept_name
                        ORDER BY occurrence_count DESC
                        """
                else:
                    # For multi-character search terms, still look for partial matches
                    # but require the term to be a complete word or phrase
                    query = f"""
                    SELECT c.concept_id, c.concept_name, COUNT(*) as occurrence_count
                    FROM {table_name} t
                    JOIN CONCEPT c ON t.{concept_id_column} = c.concept_id
                    WHERE (
                        LOWER(c.concept_name) = LOWER('{search_term}') OR
                        LOWER(c.concept_name) LIKE LOWER('% {search_term} %') OR
                        LOWER(c.concept_name) LIKE LOWER('% {search_term},%') OR
                        LOWER(c.concept_name) LIKE LOWER('{search_term} %') OR
                        LOWER(c.concept_name) LIKE LOWER('% {search_term}')
                    )
                    GROUP BY c.concept_id, c.concept_name
                    ORDER BY occurrence_count DESC
                    """
                conn = connection.connect()
                raw_conn = conn.connection
                exact_match_df = pd.read_sql(query, raw_conn)
                conn.close()
                
                if not exact_match_df.empty:
                    matching_concepts.append(exact_match_df)
            
            except Exception as e:
                pass
        
        # Combine all matching concepts
        if matching_concepts:
            all_matches_df = pd.concat(matching_concepts).drop_duplicates('concept_id')
            
            if not all_matches_df.empty:
                # Get the concept with the most occurrences
                best_match = all_matches_df.iloc[0]
                concept_id = int(best_match['concept_id'])
                concept_name = best_match['concept_name']
                occurrence_count = int(best_match['occurrence_count'])
                
                # Double-check this is a reasonable match by checking if the concept name
                # actually contains any of the search terms or their variants
                is_reasonable_match = False
                for term in search_terms:
                    # Check various forms of the search term
                    term_variants = [
                        term.lower(),
                        f" {term.lower()} ",
                        f" {term.lower()},"
                    ]
                    
                    # Special handling for potassium/k
                    if term.lower() == 'k':
                        term_variants.extend([
                            'potassium',
                            'k+',
                            'serum k'
                        ])
                    
                    # Check if any variant is in the concept name
                    for variant in term_variants:
                        if variant in concept_name.lower():
                            is_reasonable_match = True
                            break
                    
                    if is_reasonable_match:
                        break
                
                # Check if this concept actually has data
                data_check_query = f"""
                SELECT COUNT(*) as data_count
                FROM {table_name}
                WHERE {concept_id_column} = {concept_id}
                """
                conn = connection.connect()
                raw_conn = conn.connection
                data_count = pd.read_sql(data_check_query, raw_conn).iloc[0, 0]
                conn.close()
                
                if data_count > 0 and is_reasonable_match:
                    results[column_name] = {'concept_id': concept_id, 'concept_name': concept_name}
                    concept_names[column_name] = concept_name
    
    # Print only the successfully found concepts with their names
    if concept_names:
        print(f"Found {len(concept_names)} {domain.lower()} concepts:")
        for col, name in concept_names.items():
            print(f"  {col}: '{name}'")
    
    return results

# =============================================================================
# Step 4: Extract Data Using the Selected Concept IDs
# =============================================================================
def extract_person_data(connection):
    """Extract basic demographic data from the PERSON table."""
    query = """
    SELECT 
        person_id AS id_subj,
        gender_concept_id AS s_female,
        race_concept_id AS s_race,
        year_of_birth AS s_age
    FROM PERSON
    """
    try:
        conn = connection.connect()
        raw_conn = conn.connection
        
        person_df = pd.read_sql(query, raw_conn)
        
        conn.close()
        
        # If PERSON table is empty, exit the program
        if len(person_df) == 0:
            print("ERROR: PERSON table is empty. This indicates an issue with the database.")
            print("The PERSON table is the base and ground truth for this extraction.")
            print("Exiting the program.")
            import sys
            sys.exit(1)
        
        return person_df
    except Exception as e:
        print(f"Error extracting person data: {e}")
        # Return empty DataFrame instead of exiting to allow graceful failure
        return pd.DataFrame(columns=['id_subj', 's_female', 's_race', 's_age'])

def extract_measurement_data(connection, concept_id_map):
    """
    Extract measurement data using the selected concept IDs.
    
    Args:
        connection: Database connection
        concept_id_map: Dictionary mapping column names to concept IDs
    
    Returns:
        DataFrame with person_id and measurement values
    """
    if not concept_id_map:
        return pd.DataFrame()
    
    # Process each concept separately to avoid UNION ALL limit
    all_measurements = []
    
    for column_name, concept_info in concept_id_map.items():
        concept_id = concept_info['concept_id']
        
        # Query to get all measurements for each person with measurement dates
        concept_query = f"""
        SELECT
            m.person_id,
            '{column_name}' as column_name,
            m.value_as_number as value,
            m.measurement_date
        FROM MEASUREMENT m
        WHERE m.measurement_source_concept_id = {concept_id}
        AND m.value_as_number IS NOT NULL
        ORDER BY m.measurement_date DESC
        """
        
        try:
            # Execute the query using raw connection
            conn = connection.connect()
            raw_conn = conn.connection
            concept_df = pd.read_sql(concept_query, raw_conn)
            conn.close()
            
            if not concept_df.empty:
                # Keep only the most recent measurement for each person
                concept_df['measurement_date'] = pd.to_datetime(concept_df['measurement_date'])
                
                # Group by person_id and get the most recent measurement
                concept_df = concept_df.sort_values('measurement_date').groupby('person_id').first().reset_index()
                
                all_measurements.append(concept_df)
        except Exception as e:
            pass
    
    # Combine all the individual DataFrames
    if all_measurements:
        measurements_df = pd.concat(all_measurements, ignore_index=True)
        
        # Drop the measurement_date column as we don't need it anymore
        measurements_df = measurements_df.drop(columns=['measurement_date'])
        
        # Pivot the data to wide format
        pivoted_df = measurements_df.pivot(
            index='person_id',
            columns='column_name',
            values='value'
        ).reset_index()
        
        return pivoted_df
    else:
        return pd.DataFrame()

def extract_observation_data(connection, concept_id_map):
    """Extract observation data using the selected concept IDs."""
    if not concept_id_map:
        return pd.DataFrame()
    
    # Process each concept separately to avoid UNION ALL limit
    all_observations = []
    
    for column_name, concept_info in concept_id_map.items():
        concept_id = concept_info['concept_id']
        
        # Query to extract the data
        concept_query = f"""
        SELECT
            o.person_id,
            '{column_name}' as column_name,
            COALESCE(o.value_as_number, o.value_as_concept_id, 1) as value
        FROM OBSERVATION o
        WHERE o.observation_source_concept_id = {concept_id}
        """
        
        try:
            # Execute the query using raw connection
            conn = connection.connect()
            raw_conn = conn.connection
            concept_df = pd.read_sql(concept_query, raw_conn)
            conn.close()
            
            if not concept_df.empty:
                all_observations.append(concept_df)
        except Exception as e:
            pass
    
    # Combine all the individual DataFrames
    if all_observations:
        observations_df = pd.concat(all_observations, ignore_index=True)
        
        # For each person and observation type, keep only one row (doesn't matter which if there are duplicates)
        observations_df = observations_df.drop_duplicates(subset=['person_id', 'column_name'])
        
        # Pivot the data to wide format
        pivoted_df = observations_df.pivot(
            index='person_id',
            columns='column_name',
            values='value'
        ).reset_index()
        
        return pivoted_df
    else:
        return pd.DataFrame()

def extract_condition_data(connection, concept_id_map):
    """
    Extract condition data using the selected concept IDs.
    
    Args:
        connection: Database connection
        concept_id_map: Dictionary mapping column names to concept IDs
    
    Returns:
        DataFrame with person_id and condition indicators (1 if present)
    """
    if not concept_id_map:
        return pd.DataFrame()
    
    # Process each concept separately to avoid UNION ALL limit
    all_conditions = []
    
    for column_name, concept_info in concept_id_map.items():
        concept_id = concept_info['concept_id']
        
        # Query to find all patients with this condition
        query = f"""
        SELECT 
            person_id,
            '{column_name}' as column_name,
            1 as value  -- Set value to 1 to indicate presence of condition
        FROM CONDITION_OCCURRENCE
        WHERE condition_source_concept_id = {concept_id}
        GROUP BY person_id  -- Only need one record per person
        """
        
        try:
            # Execute the query using raw connection
            conn = connection.connect()
            raw_conn = conn.connection
            concept_df = pd.read_sql(query, raw_conn)
            conn.close()
            
            if not concept_df.empty:
                all_conditions.append(concept_df)
        except Exception as e:
            pass
    
    # Combine all the individual DataFrames
    if all_conditions:
        conditions_df = pd.concat(all_conditions, ignore_index=True)
        
        # Pivot the data to wide format
        pivoted_df = conditions_df.pivot(
            index='person_id',
            columns='column_name',
            values='value'
        ).reset_index()
        
        return pivoted_df
    else:
        return pd.DataFrame()

# =============================================================================
# Step 5: Main Function to Run the Pipeline
# =============================================================================
def check_database_for_columns(connection, desired_columns):
    """
    Check which desired columns might exist in the database tables directly.
    
    Args:
        connection: Database connection
        desired_columns: List of desired column names
    
    Returns:
        Dictionary with information about where columns might exist
    """
    # Get a list of all tables in the database
    tables_query = "SHOW TABLES"
    try:
        # For cursor operations, get raw connection from SQLAlchemy connection
        conn = connection.connect()
        raw_conn = conn.connection
        
        # Use pandas read_sql with raw connection
        tables = pd.read_sql(tables_query, raw_conn)
        table_names = tables.iloc[:, 0].tolist()
        
        # Check which columns exist in each table
        columns_in_tables = {}
        for table in table_names:
            try:
                columns_query = f"DESCRIBE {table}"
                columns = pd.read_sql(columns_query, raw_conn)
                column_names = columns['Field'].tolist()
                
                # Check if any of the desired columns match or are similar to columns in this table
                matching_columns = []
                for desired_col in desired_columns:
                    # Check for exact match
                    if desired_col in column_names:
                        matching_columns.append(desired_col)
                    else:
                        # Check for similar names (simplified)
                        for db_col in column_names:
                            # Remove common prefixes like id_, l_, s_
                            desired_stripped = desired_col.replace('id_', '').replace('l_', '').replace('s_', '')
                            db_stripped = db_col.replace('id_', '').replace('l_', '').replace('s_', '')
                            
                            # Check if the stripped names are similar
                            if desired_stripped.lower() == db_stripped.lower():
                                matching_columns.append((desired_col, db_col))
                
                if matching_columns:
                    columns_in_tables[table] = matching_columns
                    
            except Exception as e:
                pass
        
        # Close the connection
        conn.close()
        
        return columns_in_tables
    except Exception as e:
        print(f"Error checking database columns: {e}")
        return {}

def extract_omop_data_to_wide_format(output_file, export_tables=True):
    """
    Extract and transform OMOP CDM data.
    
    Args:
        output_file: Path to output CSV file
        export_tables: Whether to export individual tables to CSV
    """
    print("EXTRACTING DATA FROM OMOP DATABASE")
    
    try:
        # Connect to the database
        engine = connect_to_database()
        print("Database connection successful")
        
        # Check which columns might exist in the database
        column_info = check_database_for_columns(engine, DESIRED_COLUMNS)
        
        # Step 2: Find concept IDs for each feature
        measurement_concepts = find_concept_id_from_search_terms(
            engine, 'Measurement', MEASUREMENT_SEARCH_TERMS)
        
        observation_concepts = find_concept_id_from_search_terms(
            engine, 'Observation', OBSERVATION_SEARCH_TERMS)
        
        condition_concepts = find_concept_id_from_search_terms(
            engine, 'Condition', CONDITION_SEARCH_TERMS)
        
        # Step 3: Extract data using the selected concept IDs
        person_df = extract_person_data(engine)
        print(f"Extracted data for {len(person_df)} persons")
        
        measurement_df = extract_measurement_data(engine, measurement_concepts)
        observation_df = extract_observation_data(engine, observation_concepts)
        condition_df = extract_condition_data(engine, condition_concepts)
        
        # Step 4: Merge all data
        final_data = person_df.copy()
        
        # Merge measurements
        if measurement_df is not None and not measurement_df.empty:
            final_data = final_data.merge(
                measurement_df,
                left_on='id_subj',
                right_on='person_id',
                how='left'
            )
            if 'person_id' in final_data.columns:
                final_data = final_data.drop(columns=['person_id'])
        
        # Merge observations
        if not observation_df.empty:
            final_data = final_data.merge(
                observation_df, left_on='id_subj', right_on='person_id', how='left')
            if 'person_id' in final_data.columns:
                final_data = final_data.drop(columns=['person_id'])
            
            # Fill missing values with 0 for observation columns
            obs_columns = observation_df.columns.tolist()
            if 'person_id' in obs_columns:
                obs_columns.remove('person_id')
            for col in obs_columns:
                if col in final_data.columns:
                    final_data[col] = final_data[col].fillna(0)
        
        # Merge conditions
        if not condition_df.empty:
            final_data = final_data.merge(
                condition_df, left_on='id_subj', right_on='person_id', how='left')
            if 'person_id' in final_data.columns:
                final_data = final_data.drop(columns=['person_id'])
        
        # Step 5: Process the data
        # Fill indicator variables with 0
        condition_cols = [col for col in final_data.columns 
                         if col in condition_concepts.keys()]
        for col in condition_cols:
            final_data[col] = final_data[col].fillna(0).astype(int)
        
        # Fill missing values for specific columns with 0
        for col in ['l_meansbp', 'l_meandbp', 'l_ldlnew']:
            if col in final_data.columns:
                final_data[col] = final_data[col].fillna(0).astype(float)
        
        # Fill missing values for s_smoke with 0
        if 's_smoke' in final_data.columns:
            final_data['s_smoke'] = final_data['s_smoke'].fillna(0).astype(float)
        
        # Create s_smoke4 from existing s_smoke
        if 's_smoke' in final_data.columns and 's_smoke4' not in final_data.columns:
            final_data['s_smoke4'] = final_data['s_smoke'].apply(
                lambda x: 0 if x == 0 else  # Never smoker
                         1 if x == 1 else  # Former smoker
                         2 if x == 2 else  # Current smoker, some days
                         3 if x == 3 else  # Current smoker, every day
                         0)  # Changed from None to 0 for any other/missing values
        
        # Step 6: Save to CSV with only the columns that exist in our dataset
        # Only include columns that exist in the merged dataframe
        available_columns = [col for col in DESIRED_COLUMNS if col in final_data.columns]
        
        if available_columns:
            final_df = final_data[available_columns]
            
            # Get the absolute path of the output file
            abs_output_path = os.path.abspath(output_file)
            
            # Ensure header is included and properly formatted
            final_df.to_csv(output_file, index=False, header=True)
            
            print("EXTRACTION COMPLETE")
            print(f"Extracted data for {len(final_df)} patients")
            print(f"Output saved to: {abs_output_path}")
            
            # Print which columns from the desired list were not found
            missing_columns = [col for col in DESIRED_COLUMNS if col not in available_columns]
            if missing_columns:
                print(f"Columns NOT found ({len(missing_columns)}):")
                # Print in groups of 5 for readability
                for i in range(0, len(missing_columns), 5):
                    print("  " + ", ".join(missing_columns[i:i+5]))
            
            # Print which columns were successfully found
            if available_columns:
                print(f"Columns FOUND ({len(available_columns)}):")
                # Print in groups of 5 for readability
                for i in range(0, len(available_columns), 5):
                    print("  " + ", ".join(available_columns[i:i+5]))
            
            # Final message about where to find the output
            print(f"Your data has been saved to: {abs_output_path}")
            
            return final_df
        else:
            print("ERROR: No columns from the desired list were found in the data.")
            print("CSV not created.")
            return None
        
    except Exception as e:
        print(f"ERROR during extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# Step 6: Run the Script
# =============================================================================
if __name__ == "__main__":
    import sys
    import os
    
    # Default filename (always the same)
    default_filename = 'patient_data_wide.csv'
    
    # Default directory (current directory)
    output_dir = os.getcwd()
    
    # If a directory is provided as the first argument, use it
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
        
        # Make sure directory exists
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")
            except Exception as e:
                print(f"Error creating directory: {e}")
                # Fallback to current directory
                output_dir = os.getcwd()
    
    # Combine directory and filename
    output_file = os.path.join(output_dir, default_filename)
    
    # Run the extraction
    extract_omop_data_to_wide_format(output_file)
