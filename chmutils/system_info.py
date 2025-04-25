import os
import platform
import subprocess
import json
from dataclasses import dataclass, field, asdict


@dataclass
class SystemInfo:
    os_name: str = field(init=False)
    cpu_info: str = field(init=False)
    hard_drive_info: str = field(init=False)

    def __post_init__(self):
        # Determine the OS
        self.os_name = platform.system().lower()

        # Populate the CPU and hard drive info based on the OS
        if self.os_name == "windows":
            self.cpu_info = self._get_windows_cpu_info()
            self.hard_drive_info = self._get_windows_hard_drive_info()
        elif self.os_name == "linux":
            self.cpu_info = self._get_linux_cpu_info()
            self.hard_drive_info = self._get_linux_hard_drive_info()
        elif self.os_name == "darwin":  # macOS
            self.cpu_info = self._get_macos_cpu_info()
            self.hard_drive_info = self._get_macos_hard_drive_info()
        else:
            self.cpu_info = "Unsupported OS for CPU info"
            self.hard_drive_info = "Unsupported OS for hard drive info"

    def _get_windows_cpu_info(self) -> str:
        try:
            process = subprocess.Popen(['wmic', 'cpu', 'get', 'name'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, _ = process.communicate()
            return stdout.strip().split('\n')[1]  # Skip the header line
        except Exception as e:
            return f"Error retrieving CPU info: {e}"

    def _get_windows_hard_drive_info(self) -> str:
        try:
            process = subprocess.Popen(['wmic', 'diskdrive', 'get', 'model'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, _ = process.communicate()
            return ", ".join([line.strip() for line in stdout.split('\n')[1:] if line.strip()])  # Skip the header line
        except Exception as e:
            return f"Error retrieving hard drive info: {e}"

    def _get_linux_cpu_info(self) -> str:
        try:
            process = subprocess.Popen(['cat', '/proc/cpuinfo'], stdout=subprocess.PIPE, text=True)
            stdout, _ = process.communicate()
            for line in stdout.split('\n'):
                if 'model name' in line:
                    return line.split(':')[1].strip()
            return "CPU Info not found"
        except Exception as e:
            return f"Error retrieving CPU info: {e}"

    def _get_linux_hard_drive_info(self) -> str:
        try:
            process = subprocess.Popen(['lsblk', '-d', '-o', 'NAME,MODEL'], stdout=subprocess.PIPE, text=True)
            stdout, _ = process.communicate()
            return stdout.strip()
        except Exception as e:
            return f"Error retrieving hard drive info: {e}"

    def _get_macos_cpu_info(self) -> str:
        try:
            process = subprocess.Popen(['sysctl', '-n', 'machdep.cpu.brand_string'], stdout=subprocess.PIPE, text=True)
            stdout, _ = process.communicate()
            return stdout.strip()
        except Exception as e:
            return f"Error retrieving CPU info: {e}"

    def _get_macos_hard_drive_info(self) -> str:
        try:
            process = subprocess.Popen(['diskutil', 'list'], stdout=subprocess.PIPE, text=True)
            stdout, _ = process.communicate()
            return stdout.strip()
        except Exception as e:
            return f"Error retrieving hard drive info: {e}"

    def to_json(self, file_path: str = None) -> str:
        """
        Convert the system information to a JSON string.
        If a file path is provided, save the JSON string to the file.

        Args:
            file_path (str): The path to save the JSON string. Defaults to None.

        Returns:
            str: The JSON string representation of the system information.
        """
        data = asdict(self)
        json_str = json.dumps(data, indent=4)
        if file_path:
            try:
                with open(file_path, 'w') as file:
                    file.write(json_str)
            except Exception as e:
                return f"Error saving JSON to file: {e}"
        return json_str