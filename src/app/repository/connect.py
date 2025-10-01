# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
GCP-adapted database connection helper.

Purpose:
 - Remove Azure AD token logic (Azure-specific).
 - Support connecting to Cloud SQL (Postgres / MySQL) using the
   Cloud SQL Python Connector when `cloud_sql_instance_connection_name`
   is provided in config.
 - Fall back to a direct pyodbc connection (for SQL Server / MSSQL)
   using the connection string from config (useful when using the
   Cloud SQL Proxy or direct TCP access to a SQL Server instance).

How this works (assumptions / expected config fields):
 - config.graph_db_type: one of ("postgres", "mysql", "mssql"). If missing, defaults to "mssql".
 - For Cloud SQL (postgres/mysql):
     * config.cloud_sql_instance_connection_name : "<PROJECT>:<REGION>:<INSTANCE>"
     * config.db_user  : DB username
     * config.db_password : DB password (optional if using IAM DB auth or other means)
     * config.db_name  : database name
     * config.use_private_ip : optional boolean (defaults to False)
 - For MSSQL / pyodbc:
     * config.graph_db_connection_string : full pyodbc connection string (driver, server, database, uid/pwd or integrated auth)
       Typically when using Cloud SQL for SQL Server you run a cloud-sql-proxy or have a TCP endpoint and provide that in the conn string.

Notes:
 - If you plan to use Cloud SQL Connector, install: `pip install cloud-sql-python-connector[pg8000,pymysql]`
 - For MSSQL + pyodbc you must have the appropriate ODBC driver installed in the environment and provide a valid connection string.
 - This module returns a DB-API / pyodbc connection object. Close it with `.close()` when done.
"""

import logging
from typing import Optional
import os

import logger_config
from app.config import config

logger = logger_config.get_logger(__name__)

# Lazy imports for optional dependencies
_pyodbc = None
_connector = None
_ConnectorClass = None
_IPTypes = None

def _import_pyodbc():
    global _pyodbc
    if _pyodbc is None:
        try:
            import pyodbc as pyodbc_lib
        except Exception as e:
            raise RuntimeError(
                "pyodbc is required for MSSQL connections but is not installed or failed to import. "
                "Install pyodbc and ensure the system ODBC drivers are present."
            ) from e
        _pyodbc = pyodbc_lib
    return _pyodbc

def _import_connector():
    """
    Lazy-import the Cloud SQL Python Connector classes.
    """
    global _ConnectorClass, _IPTypes
    if _ConnectorClass is None:
        try:
            from google.cloud.sql.connector import Connector, IPTypes  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "cloud-sql-python-connector is required for Cloud SQL connections but is not installed or failed to import. "
                "Install it with: pip install cloud-sql-python-connector[pg8000,pymysql]"
            ) from e
        _ConnectorClass = Connector
        _IPTypes = IPTypes
    return _ConnectorClass, _IPTypes

# Reusable connector instance (recommended to keep for app lifetime)
_connector_instance: Optional["Connector"] = None


def connect():
    """
    Create and return a DB connection based on configuration.

    Returns:
        A DB-API compliant connection (pyodbc connection for mssql, DB-API for pg/mysql via connector).
    Raises:
        RuntimeError / ValueError on misconfiguration or missing dependencies.
    """
    db_type = getattr(config, "graph_db_type", None) or "mssql"
    db_type = db_type.lower()
    logger.info(f"Connecting to database (db_type={db_type})...")

    # CLOUD SQL (Postgres / MySQL) via Cloud SQL Python Connector
    if db_type in ("postgres", "mysql"):
        instance_connection_name = getattr(config, "cloud_sql_instance_connection_name", None)
        db_user = getattr(config, "db_user", None)
        db_password = getattr(config, "db_password", None)
        db_name = getattr(config, "db_name", None)
        use_private_ip = bool(getattr(config, "use_private_ip", False))

        if not instance_connection_name:
            raise ValueError(
                "For Cloud SQL connections (postgres/mysql) you must set "
                "'cloud_sql_instance_connection_name' in config to '<PROJECT>:<REGION>:<INSTANCE>'."
            )
        if not db_user:
            raise ValueError("config.db_user must be set for Cloud SQL connections.")
        if not db_name:
            raise ValueError("config.db_name must be set for Cloud SQL connections.")

        ConnectorClass, IPTypes = _import_connector()

        global _connector_instance
        if _connector_instance is None:
            # Keep a single Connector instance per process (recommended)
            _connector_instance = ConnectorClass()

        ip_type = IPTypes.PRIVATE if use_private_ip else IPTypes.PUBLIC

        if db_type == "postgres":
            # Use pg8000 (pure Python) to avoid system deps on psycopg2
            driver = "pg8000"
            try:
                conn = _connector_instance.connect(
                    instance_connection_name,
                    driver,
                    user=db_user,
                    password=db_password,
                    db=db_name,
                    ip_type=ip_type
                )
            except Exception as e:
                logger.error(f"Failed to create Postgres connection via Cloud SQL Connector: {e}")
                raise
            return conn

        else:  # mysql
            # Use pymysql driver
            driver = "pymysql"
            try:
                conn = _connector_instance.connect(
                    instance_connection_name,
                    driver,
                    user=db_user,
                    password=db_password,
                    db=db_name,
                    ip_type=ip_type
                )
            except Exception as e:
                logger.error(f"Failed to create MySQL connection via Cloud SQL Connector: {e}")
                raise
            return conn

    # MSSQL / SQL Server via pyodbc
    elif db_type in ("mssql", "sqlserver", "sql_server"):
        connection_string = getattr(config, "graph_db_connection_string", None)
        if not connection_string:
            raise ValueError(
                "For MSSQL connections please set 'graph_db_connection_string' in config. "
                "Example: 'DRIVER={ODBC Driver 18 for SQL Server};SERVER=127.0.0.1,1433;DATABASE=mydb;UID=user;PWD=pass'"
            )

        pyodbc = _import_pyodbc()
        try:
            # Connect directly; assume user manages secure connectivity (cloud-sql-proxy, private IP, firewall, etc.)
            cnxn = pyodbc.connect(connection_string)
            logger.info("Connected to MSSQL database")
            return cnxn
        except Exception as e:
            logger.error(f"Failed to connect to MSSQL via pyodbc: {e}")
            raise

    else:
        raise ValueError(f"Unsupported graph_db_type '{db_type}'. Supported values: postgres, mysql, mssql")

