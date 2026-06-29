"""
eco-logic/src/api/dell_ome.py
Dell OpenManage Enterprise REST API client.
Fetches live rack telemetry and triggers workload migration jobs.
"""

import os
import logging
import time
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class DellOMEClient:
    """
    Thin wrapper around the Dell OpenManage Enterprise REST API.

    Required environment variables:
        OME_HOST     : e.g. https://192.168.1.100
        OME_USER     : admin username
        OME_PASSWORD : admin password

    API docs: https://developer.dell.com/apis/2878/versions/4.0/openapi.yaml
    """

    AUTH_ENDPOINT = "/api/SessionService/Sessions"
    DEVICES_ENDPOINT = "/api/DeviceService/Devices"
    METRIC_ENDPOINT = "/api/MetricService/MetricData"
    JOBS_ENDPOINT = "/api/JobService/Jobs"

    def __init__(
        self,
        host: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        verify_ssl: bool = False,
        timeout: int = 30,
    ):
        self.host = (host or os.environ.get("OME_HOST", "")).rstrip("/")
        self.username = username or os.environ.get("OME_USER", "admin")
        self.password = password or os.environ.get("OME_PASSWORD", "")
        self.verify = verify_ssl
        self.timeout = timeout
        self._token: Optional[str] = None

        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503])
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.mount("http://", HTTPAdapter(max_retries=retry))

        if not verify_ssl:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # ── Auth ──────────────────────────────────────────────────────────

    def authenticate(self) -> bool:
        """Obtain a session token from OME."""
        url = self.host + self.AUTH_ENDPOINT
        payload = {
            "UserName": self.username,
            "Password": self.password,
            "SessionType": "API",
        }
        try:
            resp = self.session.post(
                url, json=payload, verify=self.verify, timeout=self.timeout
            )
            resp.raise_for_status()
            self._token = resp.headers.get("X-Auth-Token")
            self.session.headers.update(
                {"X-Auth-Token": self._token, "Content-Type": "application/json"}
            )
            logger.info(f"Authenticated with OME at {self.host}")
            return True
        except requests.RequestException as e:
            logger.error(f"OME authentication failed: {e}")
            return False

    def _ensure_auth(self):
        if not self._token:
            self.authenticate()

    # ── Device inventory ──────────────────────────────────────────────

    def get_devices(self, device_type: str = "1000") -> list:
        """
        Fetch all devices of a given type.
        Type 1000 = Servers, 2000 = Chassis, 4000 = Network devices.
        """
        self._ensure_auth()
        url = f"{self.host}{self.DEVICES_ENDPOINT}?$filter=Type eq {device_type}"
        try:
            resp = self.session.get(url, verify=self.verify, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json().get("value", [])
        except requests.RequestException as e:
            logger.error(f"get_devices failed: {e}")
            return []

    # ── Thermal & power telemetry ─────────────────────────────────────

    def get_metric_data(self, device_ids: list, metric_names: list) -> list:
        """
        Fetch live metric readings for specified devices.

        Args:
            device_ids  : List of OME device IDs
            metric_names: e.g. ["PowerConsumption", "InletTemperature", "CPUTemperature"]
        """
        self._ensure_auth()
        url = self.host + self.METRIC_ENDPOINT
        payload = {
            "DeviceIds": device_ids,
            "MetricNames": metric_names,
            "GroupType": "AllDevices",
        }
        try:
            resp = self.session.post(
                url, json=payload, verify=self.verify, timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json().get("value", [])
        except requests.RequestException as e:
            logger.error(f"get_metric_data failed: {e}")
            return []

    def get_rack_telemetry(self, rack_device_ids: list) -> dict:
        """
        High-level helper: return dict of {device_id: {temp, power, coolant_flow}}.
        """
        raw = self.get_metric_data(
            device_ids=rack_device_ids,
            metric_names=["InletTemperature", "PowerConsumption", "CoolantFlowRate"],
        )
        result = {}
        for entry in raw:
            dev_id = entry.get("DeviceId")
            if dev_id not in result:
                result[dev_id] = {"temp": None, "power": None, "coolant_flow": None}
            name = entry.get("MetricName", "")
            val = entry.get("Value")
            if name == "InletTemperature":
                result[dev_id]["temp"] = float(val)
            if name == "PowerConsumption":
                result[dev_id]["power"] = float(val)
            if name == "CoolantFlowRate":
                result[dev_id]["coolant_flow"] = float(val)
        return result

    # ── Workload orchestration via Jobs API ───────────────────────────

    def trigger_workload_migration(
        self,
        source_rack_id: int,
        target_rack_id: int,
        workload_name: str,
        reason: str = "RL thermal optimization",
    ) -> Optional[int]:
        """
        Submit a workload migration job to OME.
        Returns the job ID, or None on failure.
        """
        self._ensure_auth()
        url = self.host + self.JOBS_ENDPOINT
        payload = {
            "JobName": f"EcoLogic-Migrate-{workload_name}",
            "JobDescription": reason,
            "Schedule": "startnow",
            "State": "Enabled",
            "JobType": {"Id": 5, "Name": "Update_Task"},
            "Params": [
                {"Key": "source_rack", "Value": str(source_rack_id)},
                {"Key": "target_rack", "Value": str(target_rack_id)},
                {"Key": "workload_name", "Value": workload_name},
                {"Key": "optimization", "Value": "thermal_rl"},
            ],
            "Targets": [{"Id": source_rack_id, "Type": {"Id": 1000, "Name": "DEVICE"}}],
        }
        try:
            resp = self.session.post(
                url, json=payload, verify=self.verify, timeout=self.timeout
            )
            resp.raise_for_status()
            job_id = resp.json().get("Id")
            logger.info(
                f"Migration job {job_id} submitted: {workload_name} R{source_rack_id}→R{target_rack_id}"
            )
            return job_id
        except requests.RequestException as e:
            logger.error(f"trigger_workload_migration failed: {e}")
            return None

    def get_job_status(self, job_id: int) -> Optional[str]:
        """Poll job status. Returns 'Running', 'Completed', 'Failed', etc."""
        self._ensure_auth()
        url = f"{self.host}{self.JOBS_ENDPOINT}({job_id})"
        try:
            resp = self.session.get(url, verify=self.verify, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json().get("LastRunStatus", {}).get("Name")
        except requests.RequestException as e:
            logger.error(f"get_job_status failed: {e}")
            return None

    def wait_for_job(
        self, job_id: int, poll_interval: int = 10, max_wait: int = 300
    ) -> str:
        """Block until job completes or times out."""
        elapsed = 0
        while elapsed < max_wait:
            status = self.get_job_status(job_id)
            if status in ("Completed", "Failed", "Warning"):
                return status
            time.sleep(poll_interval)
            elapsed += poll_interval
        return "Timeout"
