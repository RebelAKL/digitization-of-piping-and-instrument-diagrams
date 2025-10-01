# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from pydantic import BaseSettings, root_validator, validator
from typing import Union, Optional, Set


class Config(BaseSettings):
    # --- Functional / domain settings (unchanged) ---
    arrow_symbol_label: str = 'Piping/Fittings/Mid arrow flow direction'
    centroid_distance_threshold: float = 0.5
    debug: bool = False
    detect_dotted_lines: bool = False
    enable_preprocessing_text_detection: bool = True
    enable_thinning_preprocessing_line_detection: bool = True
    flow_direction_asset_prefixes: Union[str, Set[str]] = \
        {'Equipment/', 'Piping/Endpoint/Pagination'}
    graph_distance_threshold_for_lines_pixels: int = 50
    graph_distance_threshold_for_symbols_pixels: int = 5
    graph_distance_threshold_for_text_pixels: int = 5
    graph_line_buffer_pixels: int = 5
    graph_symbol_to_symbol_distance_threshold_pixels: int = 10
    graph_symbol_to_symbol_overlap_region_threshold: float = 0.7
    inference_score_threshold: float = 0.5
    inference_service_retry_count: int = 3
    inference_service_retry_backoff_factor: float = 0.3
    line_detection_hough_max_line_gap: Optional[int] = None
    line_detection_hough_min_line_length: Optional[int] = 10
    # line_detection_hough_max_line_gap value helps with returning the smaller dashed line segments...
    line_detection_hough_rho: float = 0.1
    line_detection_hough_theta: int = 1080
    line_detection_hough_threshold: int = 5
    line_detection_job_timeout_seconds: int = 300
    line_segment_padding_default: float = 0.2

    port: int = 8000
    symbol_detection_api: str = str()
    symbol_detection_api_bearer_token: str = str()
    symbol_label_prefixes_to_connect_if_close: Union[str, Set[str]] = \
        {'Equipment', 'Instrument/Valve/', 'Piping/Fittings/Mid arrow flow direction', 'Piping/Fittings/Flanged connection'}
    symbol_label_prefixes_to_include_in_graph_image_output: Union[str, Set[str]] = \
        {'Equipment/', 'Instrument/Valve/', 'Piping/Endpoint/Pagination'}
    symbol_label_prefixes_with_text: Union[str, Set[str]] = \
        {'Equipment/', 'Instrument/', 'Piping/Endpoint/Pagination'}
    symbol_overlap_threshold: float = 0.6
    text_detection_area_intersection_ratio_threshold: float = 0.8
    text_detection_distance_threshold: float = 0.01
    symbol_label_for_connectors: Union[str, Set[str]] = \
        {'Piping/Endpoint/Pagination'}
    valve_symbol_prefix: str = 'Instrument/Valve/'
    workers_count_for_data_batch: int = 3

    # --- Storage (Azure compatibility + GCP) ---
    # Azure-compatible names (kept for backward compatibility)
    blob_storage_account_url: str = str()
    blob_storage_container_name: str = str()

    # GCP equivalents (preferred for GCP migration)
    gcs_bucket_name: Optional[str] = None
    gcp_project: Optional[str] = None

    # --- OCR / form recognizer (kept but optional) ---
    # Original Azure form recognizer endpoint (kept for compatibility; not required for GCP Vision)
    form_recognizer_endpoint: Optional[str] = None

    # --- Database (Azure compatibility + GCP Cloud SQL) ---
    # Keep original connection string support (e.g., for SQL Server via pyodbc)
    graph_db_connection_string: str = str()

    # Remove Azure AD toggle — for GCP we use Cloud SQL / direct connection
    # New fields for Cloud SQL (postgres/mysql)
    cloud_sql_instance_connection_name: Optional[str] = None  # "<PROJECT>:<REGION>:<INSTANCE>"
    db_user: Optional[str] = None
    db_password: Optional[str] = None
    db_name: Optional[str] = None
    graph_db_type: str = "mssql"  # one of: "mssql", "postgres", "mysql"
    use_private_ip: bool = False

    # --- Other cloud / integration fields (keep originals for compatibility) ---
    # Note: symbol_detection_api and symbol_detection_api_bearer_token retained above
    # You can add Document AI processor config later; kept out for now for generality.

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        # Allow case-insensitive env names if you like:
        # env_prefix = ''

    # --- Validators / transformations ---

    @validator(
        "symbol_detection_api",
        "symbol_detection_api_bearer_token",
        pre=True,
        allow_reuse=True
    )
    def validate_string_non_empty_for_symbol_api(cls, v, field):
        """
        For symbol detection API integration we keep these required (non-empty).
        If you want them optional during local unit tests, set them via env or update this validator.
        """
        if v is None or (isinstance(v, str) and len(v.strip()) == 0):
            raise ValueError(f"Value for '{field.name}' must be a non-empty string")
        return v

    @validator(
        "flow_direction_asset_prefixes",
        "symbol_label_prefixes_with_text",
        "symbol_label_prefixes_to_include_in_graph_image_output",
        "symbol_label_prefixes_to_connect_if_close",
        pre=True,
        allow_reuse=True
    )
    def validate_and_transform_comma_separated_list(cls, val):
        if isinstance(val, str):
            val_arr = val.split(',')
            val_arr = [x.strip() for x in val_arr if x.strip() != '']
            return set(val_arr)
        return val

    @root_validator(allow_reuse=True)
    def validate_storage_and_db_configuration(cls, values):
        """
        Validate storage and database configuration in a backward-compatible way:
         - Require at least one storage: gcs_bucket_name OR blob_storage_container_name
         - Require at least one DB config: cloud_sql_instance_connection_name (Cloud SQL) OR graph_db_connection_string (pyodbc / MSSQL)
         - Keep previous dotted-lines root logic for line Hough parameters
        """
        # --- Storage check ---
        gcs_bucket = values.get('gcs_bucket_name')
        blob_container = values.get('blob_storage_container_name')

        if not gcs_bucket and (blob_container is None or len(str(blob_container).strip()) == 0):
            # if no storage configured, fail fast — pipeline requires a storage target for outputs
            raise ValueError("Configuration error: either 'gcs_bucket_name' or 'blob_storage_container_name' must be provided")

        # --- DB check ---
        cloud_sql_instance = values.get('cloud_sql_instance_connection_name')
        graph_conn_str = values.get('graph_db_connection_string')

        if (cloud_sql_instance is None or len(str(cloud_sql_instance).strip()) == 0) and (graph_conn_str is None or len(str(graph_conn_str).strip()) == 0):
            raise ValueError("Configuration error: either 'cloud_sql_instance_connection_name' (Cloud SQL) or 'graph_db_connection_string' (MSSQL) must be provided")

        # If graph_db_type provided, normalize to lower-case and ensure expected values
        graph_db_type = values.get('graph_db_type') or 'mssql'
        graph_db_type = str(graph_db_type).lower()
        if graph_db_type not in ('mssql', 'postgres', 'mysql'):
            raise ValueError("Unsupported 'graph_db_type'. Supported values: 'mssql', 'postgres', 'mysql'")
        values['graph_db_type'] = graph_db_type

        # --- dotted lines handling (preserve original logic) ---
        detect_dotted = values.get('detect_dotted_lines', False)
        if detect_dotted is True:
            values['line_detection_hough_min_line_length'] = None
            if values.get('line_detection_hough_max_line_gap') is None:
                values['line_detection_hough_max_line_gap'] = 10
        else:
            if values.get('line_detection_hough_min_line_length') is None or values.get('line_detection_hough_min_line_length') < 10:
                values['line_detection_hough_min_line_length'] = 10
            values['line_detection_hough_max_line_gap'] = None

        return values


# Instantiate configuration
config = Config()
