
from app.config import Config, config
from typing import Optional, Union, Any
from logger_config import get_logger

import google.auth
from google.cloud import storage
from google.api_core.exceptions import NotFound

logger = get_logger(__name__)


class BlobStorageClient:
    """
    GCP-backed replacement for the Azure BlobStorageClient.
    Public API preserved: init(), upload_bytes(), download_bytes(), blob_exists().

    Mapping:
      - Azure container -> GCS bucket
      - Azure blob -> GCS object
    """

    _bucket: Optional[storage.Bucket] = None
    _client: Optional[storage.Client] = None

    def __init__(self, config: Config = config, credential: Optional[Any] = None):
        """
        :param config: configuration object. Expected keys:
                       - gcs_bucket_name (preferred)
                       - OR blob_storage_container_name (falls back to this for compatibility)
                       - optionally: gcp_project
        :param credential: optional google.auth credentials object.
                           If None, Application Default Credentials (ADC) are used.
        """
        self._config = config
        self._credential = credential

    def throw_if_not_initialized(self):
        """Throws an exception if the storage client is not initialized."""
        if self._bucket is None or self._client is None:
            raise Exception('Blob storage client is not initialized')

    def upload_bytes(self, blob_name: str, image_bytes: Union[bytes, str]):
        """
        Uploads the given bytes/string to the configured GCS bucket.

        :param blob_name: destination object name in the bucket
        :param image_bytes: bytes or string to upload
        :return: the google.cloud.storage.Blob instance
        """
        logger.info(f'Uploading {blob_name} to GCS bucket')

        self.throw_if_not_initialized()
        blob = self._bucket.blob(blob_name)

        # upload_from_string handles both bytes and text
        if isinstance(image_bytes, (bytes, bytearray)):
            blob.upload_from_string(image_bytes)
        else:
            blob.upload_from_string(str(image_bytes))

        # Returning the blob keeps return type flexible like Azure client did.
        return blob

    def download_bytes(self, blob_name: str) -> bytes:
        """
        Downloads the given object from the GCS bucket and returns bytes.

        :param blob_name: the object name
        :return: bytes of the object
        """
        logger.info(f'Downloading {blob_name} from GCS bucket')

        self.throw_if_not_initialized()
        blob = self._bucket.blob(blob_name)

        if not blob.exists(client=self._client):
            raise FileNotFoundError(f"GCS object '{blob_name}' not found in bucket '{self._bucket.name}'")

        return blob.download_as_bytes()

    def blob_exists(self, blob_name: str) -> bool:
        """
        Checks if the given object exists in the GCS bucket.

        :param blob_name: the object name to check
        :return: True if exists, False otherwise
        """
        logger.info(f'Checking if {blob_name} exists in GCS bucket')

        self.throw_if_not_initialized()
        blob = self._bucket.blob(blob_name)
        return blob.exists(client=self._client)

    def init(self):
        """
        Initializes the GCS client and loads the bucket specified in the config.

        Expects either:
          - config.gcs_bucket_name (preferred)
          - OR config.blob_storage_container_name (for compatibility)

        Optionally uses config.gcp_project or falls back to ADC project.
        """
        # Resolve bucket name (support old config key for easier migration)
        bucket_name = getattr(self._config, "gcs_bucket_name", None) or \
                      getattr(self._config, "blob_storage_container_name", None)
        if not bucket_name:
            raise ValueError("config must provide 'gcs_bucket_name' or 'blob_storage_container_name'")

        # Use provided credential or ADC
        creds = self._credential
        project = getattr(self._config, "gcp_project", None)

        if creds is None:
            creds, adc_project = google.auth.default()
            if project is None:
                project = adc_project

        # Create storage client (project optional)
        if project:
            self._client = storage.Client(project=project, credentials=creds)
        else:
            self._client = storage.Client(credentials=creds)

        try:
            # get_bucket will raise NotFound if missing
            self._bucket = self._client.get_bucket(bucket_name)
        except NotFound:
            raise FileNotFoundError(f"Bucket '{bucket_name}' not found in project '{project}'")


# Default instance pattern (similar to original)
blob_storage_client = BlobStorageClient(config)
# NOTE: call blob_storage_client.init() during app startup after config/env are ready.
