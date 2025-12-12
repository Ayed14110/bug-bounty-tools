#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║     █████╗ ██╗   ██╗███████╗██████╗     ██╗   ██╗ ██╗███████╗       █████╗ ██╗    ██████╗  ██████╗ ██╗    ██╗███████╗██████╗   ║
║    ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗    ██║   ██║███║██╔════╝      ██╔══██╗██║    ██╔══██╗██╔═══██╗██║    ██║██╔════╝██╔══██╗  ║
║    ███████║ ╚████╔╝ █████╗  ██║  ██║    ██║   ██║╚██║███████╗█████╗███████║██║    ██████╔╝██║   ██║██║ █╗ ██║█████╗  ██████╔╝  ║
║    ██╔══██║  ╚██╔╝  ██╔══╝  ██║  ██║    ╚██╗ ██╔╝ ██║╚════██║╚════╝██╔══██║██║    ██╔═══╝ ██║   ██║██║███╗██║██╔══╝  ██╔══██╗  ║
║    ██║  ██║   ██║   ███████╗██████╔╝     ╚████╔╝  ██║███████║      ██║  ██║██║    ██║     ╚██████╔╝╚███╔███╔╝███████╗██║  ██║  ║
║    ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═════╝       ╚═══╝   ╚═╝╚══════╝      ╚═╝  ╚═╝╚═╝    ╚═╝      ╚═════╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝  ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║           ★ VERSION 15.0 AI-POWERED - 500+ TOOLS + 3 OLLAMA MODELS + SMART AUTO-INSTALLER ★                                   ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  🤖 AI MODELS: DeepSeek-R1 │ Llama3.3 │ Codestral │ Qwen2.5-Coder │ Mistral                                                   ║
║  🔧 SMART INSTALLER: Auto-detect Kali/Termux/Ubuntu/Debian/Arch/Fedora/macOS                                                  ║
║  ⚡ AI ASSISTANCE: Every step, every option, every recommendation powered by AI                                               ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

VERSION = "15.0 AI-POWERED"
import os, sys, re, json, time, socket, subprocess, shutil, hashlib, platform
import threading, signal, struct, urllib.request, ssl, http.client
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urljoin, quote
import random
import string

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM DETECTION - INTELLIGENT OS RECOGNITION
# ═══════════════════════════════════════════════════════════════════════════════
class SystemType(Enum):
    KALI = "kali"
    PARROT = "parrot"
    BLACKARCH = "blackarch"
    TERMUX = "termux"
    UBUNTU = "ubuntu"
    DEBIAN = "debian"
    ARCH = "arch"
    FEDORA = "fedora"
    CENTOS = "centos"
    RHEL = "rhel"
    MACOS = "macos"
    WINDOWS = "windows"
    UNKNOWN = "unknown"

class SystemDetector:
    """Intelligent System Detection"""
    
    @staticmethod
    def detect() -> Dict:
        info = {
            "type": SystemType.UNKNOWN,
            "name": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "is_root": os.geteuid() == 0 if hasattr(os, 'geteuid') else False,
            "home": str(Path.home()),
            "package_manager": None,
            "install_cmd": None,
            "python": sys.executable,
            "shell": os.environ.get('SHELL', '/bin/bash')
        }
        
        # Detect Termux first
        if 'com.termux' in str(Path.home()) or os.path.exists('/data/data/com.termux'):
            info["type"] = SystemType.TERMUX
            info["package_manager"] = "pkg"
            info["install_cmd"] = "pkg install -y"
            info["name"] = "Termux (Android)"
            return info
        
        # Check Linux distributions
        if platform.system() == "Linux":
            # Read os-release
            os_release = {}
            for path in ['/etc/os-release', '/etc/lsb-release']:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        for line in f:
                            if '=' in line:
                                key, val = line.strip().split('=', 1)
                                os_release[key] = val.strip('"')
            
            distro_id = os_release.get('ID', '').lower()
            distro_like = os_release.get('ID_LIKE', '').lower()
            
            # Kali Linux
            if 'kali' in distro_id:
                info["type"] = SystemType.KALI
                info["package_manager"] = "apt"
                info["install_cmd"] = "apt install -y"
                info["name"] = os_release.get('PRETTY_NAME', 'Kali Linux')
            
            # Parrot OS
            elif 'parrot' in distro_id:
                info["type"] = SystemType.PARROT
                info["package_manager"] = "apt"
                info["install_cmd"] = "apt install -y"
                info["name"] = os_release.get('PRETTY_NAME', 'Parrot OS')
            
            # BlackArch
            elif 'blackarch' in distro_id or os.path.exists('/etc/blackarch-release'):
                info["type"] = SystemType.BLACKARCH
                info["package_manager"] = "pacman"
                info["install_cmd"] = "pacman -S --noconfirm"
                info["name"] = "BlackArch Linux"
            
            # Ubuntu
            elif 'ubuntu' in distro_id:
                info["type"] = SystemType.UBUNTU
                info["package_manager"] = "apt"
                info["install_cmd"] = "apt install -y"
                info["name"] = os_release.get('PRETTY_NAME', 'Ubuntu')
            
            # Debian
            elif 'debian' in distro_id or 'debian' in distro_like:
                info["type"] = SystemType.DEBIAN
                info["package_manager"] = "apt"
                info["install_cmd"] = "apt install -y"
                info["name"] = os_release.get('PRETTY_NAME', 'Debian')
            
            # Arch Linux
            elif 'arch' in distro_id:
                info["type"] = SystemType.ARCH
                info["package_manager"] = "pacman"
                info["install_cmd"] = "pacman -S --noconfirm"
                info["name"] = os_release.get('PRETTY_NAME', 'Arch Linux')
            
            # Fedora
            elif 'fedora' in distro_id:
                info["type"] = SystemType.FEDORA
                info["package_manager"] = "dnf"
                info["install_cmd"] = "dnf install -y"
                info["name"] = os_release.get('PRETTY_NAME', 'Fedora')
            
            # CentOS/RHEL
            elif 'centos' in distro_id or 'rhel' in distro_id:
                info["type"] = SystemType.CENTOS
                info["package_manager"] = "yum"
                info["install_cmd"] = "yum install -y"
                info["name"] = os_release.get('PRETTY_NAME', 'CentOS/RHEL')
        
        # macOS
        elif platform.system() == "Darwin":
            info["type"] = SystemType.MACOS
            info["package_manager"] = "brew"
            info["install_cmd"] = "brew install"
            info["name"] = f"macOS {platform.mac_ver()[0]}"
        
        # Windows
        elif platform.system() == "Windows":
            info["type"] = SystemType.WINDOWS
            info["package_manager"] = "choco"
            info["install_cmd"] = "choco install -y"
            info["name"] = f"Windows {platform.release()}"
        
        return info

# ═══════════════════════════════════════════════════════════════════════════════
# OLLAMA AI INTEGRATION - 3 POWERFUL MODELS
# ═══════════════════════════════════════════════════════════════════════════════
class AIModel(Enum):
    DEEPSEEK_R1 = "deepseek-r1:latest"           # Best for reasoning & analysis
    LLAMA3 = "llama3.3:latest"                   # Best for general tasks
    CODESTRAL = "codestral:latest"               # Best for code & security
    QWEN_CODER = "qwen2.5-coder:latest"         # Alternative coder
    MISTRAL = "mistral:latest"                   # Fast & efficient

class OllamaAI:
    """Ollama AI Integration with Multiple Models"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.available_models = []
        self.current_model = None
        self.is_available = False
        self.conversation_history = []
    
    def check_status(self) -> bool:
        """Check if Ollama is running"""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.base_url}/api/tags", method='GET')
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                self.available_models = [m['name'] for m in data.get('models', [])]
                self.is_available = True
                return True
        except Exception as e:
            self.is_available = False
            return False
    
    def list_models(self) -> List[str]:
        """List available models"""
        self.check_status()
        return self.available_models
    
    def pull_model(self, model: str) -> bool:
        """Pull/download a model"""
        try:
            print(f"    {C.YEL}[AI]{C.R} Downloading {model}... (this may take a while)")
            result = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True, text=True, timeout=3600
            )
            return result.returncode == 0
        except Exception as e:
            print(f"    {C.RED}[!]{C.R} Failed to pull model: {e}")
            return False
    
    def set_model(self, model: AIModel):
        """Set the current model"""
        model_name = model.value if isinstance(model, AIModel) else model
        if model_name in self.available_models:
            self.current_model = model_name
            return True
        return False
    
    def chat(self, prompt: str, model: str = None, system_prompt: str = None, 
             stream: bool = False, context: str = None) -> str:
        """Send a chat message to Ollama"""
        if not self.is_available:
            return "[AI Offline] Ollama is not available"
        
        model = model or self.current_model or "llama3.3:latest"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if context:
            messages.append({"role": "user", "content": f"Context: {context}"})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            data = json.dumps({
                "model": model,
                "messages": messages,
                "stream": stream
            }).encode('utf-8')
            
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=data,
                headers={"Content-Type": "application/json"},
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode())
                return result.get('message', {}).get('content', 'No response')
        except Exception as e:
            return f"[AI Error] {str(e)}"
    
    def analyze_target(self, target: str) -> str:
        """AI analyzes target and suggests attack vectors"""
        system = """You are a professional penetration tester AI assistant. 
        Analyze the target and provide:
        1. Potential attack vectors
        2. Recommended tools to use
        3. Testing methodology
        4. Risk assessment
        Be concise and professional."""
        
        return self.chat(
            f"Analyze this target for security testing: {target}",
            system_prompt=system
        )
    
    def analyze_results(self, tool: str, output: str) -> str:
        """AI analyzes tool output for vulnerabilities"""
        system = """You are a security analyst AI. Analyze tool output and provide:
        1. Key findings and vulnerabilities
        2. Severity assessment (Critical/High/Medium/Low)
        3. Exploitation possibilities
        4. Remediation recommendations
        Be concise and actionable."""
        
        return self.chat(
            f"Analyze this {tool} output:\n{output[:3000]}",  # Limit output size
            system_prompt=system
        )
    
    def suggest_next_step(self, current_results: List[Dict]) -> str:
        """AI suggests next testing steps based on results"""
        system = """You are a penetration testing strategist. Based on current findings:
        1. Suggest the next logical testing steps
        2. Identify potential attack chains
        3. Recommend specific tools and techniques
        Be strategic and methodical."""
        
        summary = "\n".join([f"- {r.get('tool')}: {r.get('status')}" for r in current_results[:20]])
        return self.chat(
            f"Based on these results, suggest next steps:\n{summary}",
            system_prompt=system
        )
    
    def generate_report_summary(self, results: List[Dict]) -> str:
        """AI generates executive summary"""
        system = """You are a cybersecurity report writer. Generate a professional 
        executive summary including:
        1. Overall security posture assessment
        2. Critical findings summary
        3. Key recommendations
        4. Risk rating
        Be professional and concise."""
        
        findings = "\n".join([
            f"- {r.get('tool')}: {r.get('status')} - {r.get('output', '')[:200]}"
            for r in results if r.get('status') == 'success'
        ][:30])
        
        return self.chat(
            f"Generate executive summary for these findings:\n{findings}",
            system_prompt=system
        )
    
    def explain_tool(self, tool_name: str) -> str:
        """AI explains what a tool does"""
        system = """You are a cybersecurity educator. Explain the tool including:
        1. What it does
        2. Common use cases
        3. Key options/flags
        4. Example usage
        Be educational and clear."""
        
        return self.chat(f"Explain the security tool: {tool_name}", system_prompt=system)
    
    def fix_error(self, tool: str, error: str) -> str:
        """AI helps fix tool errors"""
        system = """You are a troubleshooting expert. For the given error:
        1. Explain what went wrong
        2. Provide specific fix commands
        3. Suggest alternatives if needed
        Be practical and solution-oriented."""
        
        return self.chat(
            f"Tool '{tool}' failed with error: {error}\nHow to fix?",
            system_prompt=system
        )

# ═══════════════════════════════════════════════════════════════════════════════
# SMART INSTALLER - INTELLIGENT TOOL INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════════
class SmartInstaller:
    """Intelligent Multi-Platform Tool Installer"""
    
    # Tool mappings for different systems
    TOOL_PACKAGES = {
        # Format: tool_name: {system_type: package_name or [packages]}
        "nmap": {
            SystemType.KALI: "nmap",
            SystemType.PARROT: "nmap",
            SystemType.UBUNTU: "nmap",
            SystemType.DEBIAN: "nmap",
            SystemType.ARCH: "nmap",
            SystemType.FEDORA: "nmap",
            SystemType.TERMUX: "nmap",
            SystemType.MACOS: "nmap",
        },
        "masscan": {
            SystemType.KALI: "masscan",
            SystemType.UBUNTU: "masscan",
            SystemType.DEBIAN: "masscan",
            SystemType.ARCH: "masscan",
            SystemType.TERMUX: "masscan",
            SystemType.MACOS: "masscan",
        },
        "nikto": {
            SystemType.KALI: "nikto",
            SystemType.UBUNTU: "nikto",
            SystemType.DEBIAN: "nikto",
            SystemType.ARCH: "nikto",
        },
        "sqlmap": {
            SystemType.KALI: "sqlmap",
            SystemType.UBUNTU: "sqlmap",
            SystemType.DEBIAN: "sqlmap",
            SystemType.ARCH: "sqlmap",
            SystemType.TERMUX: "sqlmap",
        },
        "hydra": {
            SystemType.KALI: "hydra",
            SystemType.UBUNTU: "hydra",
            SystemType.DEBIAN: "hydra",
            SystemType.ARCH: "hydra",
            SystemType.TERMUX: "hydra",
        },
        "john": {
            SystemType.KALI: "john",
            SystemType.UBUNTU: "john",
            SystemType.DEBIAN: "john",
            SystemType.ARCH: "john",
            SystemType.TERMUX: "john",
        },
        "hashcat": {
            SystemType.KALI: "hashcat",
            SystemType.UBUNTU: "hashcat",
            SystemType.DEBIAN: "hashcat",
            SystemType.ARCH: "hashcat",
        },
        "gobuster": {
            SystemType.KALI: "gobuster",
            SystemType.UBUNTU: "gobuster",
            SystemType.ARCH: "gobuster",
        },
        "ffuf": {
            SystemType.KALI: "ffuf",
            SystemType.ARCH: "ffuf",
        },
        "nuclei": {
            "pip": "nuclei",
            "go": "github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest",
        },
        "subfinder": {
            "go": "github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest",
        },
        "httpx": {
            "go": "github.com/projectdiscovery/httpx/cmd/httpx@latest",
        },
        "amass": {
            SystemType.KALI: "amass",
            SystemType.UBUNTU: "amass",
            "snap": "amass",
        },
        "theHarvester": {
            SystemType.KALI: "theharvester",
            SystemType.UBUNTU: "theharvester",
            "pip": "theHarvester",
        },
        "volatility": {
            SystemType.KALI: "volatility3",
            "pip": "volatility3",
        },
        "foremost": {
            SystemType.KALI: "foremost",
            SystemType.UBUNTU: "foremost",
            SystemType.DEBIAN: "foremost",
        },
        "binwalk": {
            SystemType.KALI: "binwalk",
            SystemType.UBUNTU: "binwalk",
            SystemType.DEBIAN: "binwalk",
            SystemType.TERMUX: "binwalk",
            "pip": "binwalk",
        },
        "exiftool": {
            SystemType.KALI: "libimage-exiftool-perl",
            SystemType.UBUNTU: "libimage-exiftool-perl",
            SystemType.DEBIAN: "libimage-exiftool-perl",
            SystemType.TERMUX: "exiftool",
        },
        "wireshark": {
            SystemType.KALI: "wireshark",
            SystemType.UBUNTU: "wireshark",
            SystemType.DEBIAN: "wireshark",
        },
        "tcpdump": {
            SystemType.KALI: "tcpdump",
            SystemType.UBUNTU: "tcpdump",
            SystemType.DEBIAN: "tcpdump",
            SystemType.TERMUX: "tcpdump",
        },
        "netcat": {
            SystemType.KALI: "netcat-traditional",
            SystemType.UBUNTU: "netcat",
            SystemType.DEBIAN: "netcat",
            SystemType.TERMUX: "netcat-openbsd",
        },
        "curl": {
            SystemType.KALI: "curl",
            SystemType.UBUNTU: "curl",
            SystemType.DEBIAN: "curl",
            SystemType.TERMUX: "curl",
        },
        "wget": {
            SystemType.KALI: "wget",
            SystemType.UBUNTU: "wget",
            SystemType.DEBIAN: "wget",
            SystemType.TERMUX: "wget",
        },
        "git": {
            SystemType.KALI: "git",
            SystemType.UBUNTU: "git",
            SystemType.DEBIAN: "git",
            SystemType.TERMUX: "git",
            SystemType.ARCH: "git",
        },
        "python3": {
            SystemType.KALI: "python3",
            SystemType.UBUNTU: "python3",
            SystemType.DEBIAN: "python3",
            SystemType.TERMUX: "python",
        },
        "pip": {
            SystemType.KALI: "python3-pip",
            SystemType.UBUNTU: "python3-pip",
            SystemType.DEBIAN: "python3-pip",
        },
        "go": {
            SystemType.KALI: "golang",
            SystemType.UBUNTU: "golang",
            SystemType.DEBIAN: "golang",
            SystemType.TERMUX: "golang",
            SystemType.ARCH: "go",
        },
        "ruby": {
            SystemType.KALI: "ruby",
            SystemType.UBUNTU: "ruby",
            SystemType.DEBIAN: "ruby",
            SystemType.TERMUX: "ruby",
        },
        "metasploit": {
            SystemType.KALI: "metasploit-framework",
        },
        "burpsuite": {
            SystemType.KALI: "burpsuite",
        },
        "aircrack-ng": {
            SystemType.KALI: "aircrack-ng",
            SystemType.UBUNTU: "aircrack-ng",
            SystemType.TERMUX: "aircrack-ng",
        },
        "wifite": {
            SystemType.KALI: "wifite",
        },
        "responder": {
            SystemType.KALI: "responder",
        },
        "crackmapexec": {
            SystemType.KALI: "crackmapexec",
            "pip": "crackmapexec",
        },
        "impacket": {
            SystemType.KALI: "python3-impacket",
            "pip": "impacket",
        },
        "bloodhound": {
            SystemType.KALI: "bloodhound",
            "pip": "bloodhound",
        },
        "enum4linux": {
            SystemType.KALI: "enum4linux",
        },
        "smbclient": {
            SystemType.KALI: "smbclient",
            SystemType.UBUNTU: "smbclient",
            SystemType.DEBIAN: "smbclient",
        },
        "lynis": {
            SystemType.KALI: "lynis",
            SystemType.UBUNTU: "lynis",
            SystemType.DEBIAN: "lynis",
        },
        "rkhunter": {
            SystemType.KALI: "rkhunter",
            SystemType.UBUNTU: "rkhunter",
            SystemType.DEBIAN: "rkhunter",
        },
        "chkrootkit": {
            SystemType.KALI: "chkrootkit",
            SystemType.UBUNTU: "chkrootkit",
            SystemType.DEBIAN: "chkrootkit",
        },
        "clamav": {
            SystemType.KALI: "clamav",
            SystemType.UBUNTU: "clamav",
            SystemType.DEBIAN: "clamav",
            SystemType.TERMUX: "clamav",
        },
        "snort": {
            SystemType.KALI: "snort",
            SystemType.UBUNTU: "snort",
        },
        "suricata": {
            SystemType.KALI: "suricata",
            SystemType.UBUNTU: "suricata",
        },
        "fail2ban": {
            SystemType.KALI: "fail2ban",
            SystemType.UBUNTU: "fail2ban",
            SystemType.DEBIAN: "fail2ban",
        },
        "wpscan": {
            SystemType.KALI: "wpscan",
            "gem": "wpscan",
        },
        "whatweb": {
            SystemType.KALI: "whatweb",
            SystemType.UBUNTU: "whatweb",
        },
        "dirb": {
            SystemType.KALI: "dirb",
            SystemType.UBUNTU: "dirb",
        },
        "dirsearch": {
            "pip": "dirsearch",
            "git": "https://github.com/maurosoria/dirsearch.git",
        },
        "xsstrike": {
            "git": "https://github.com/s0md3v/XSStrike.git",
            "pip": "xsstrike",
        },
        "shodan": {
            "pip": "shodan",
        },
        "censys": {
            "pip": "censys",
        },
        "maltego": {
            SystemType.KALI: "maltego",
        },
        "recon-ng": {
            SystemType.KALI: "recon-ng",
            "pip": "recon-ng",
        },
        "sherlock": {
            "pip": "sherlock-project",
            "git": "https://github.com/sherlock-project/sherlock.git",
        },
        "spiderfoot": {
            "pip": "spiderfoot",
        },
        "photon": {
            "git": "https://github.com/s0md3v/Photon.git",
        },
        "wafw00f": {
            SystemType.KALI: "wafw00f",
            "pip": "wafw00f",
        },
        "arjun": {
            "pip": "arjun",
        },
        "paramspider": {
            "pip": "paramspider",
        },
        "gau": {
            "go": "github.com/lc/gau/v2/cmd/gau@latest",
        },
        "waybackurls": {
            "go": "github.com/tomnomnom/waybackurls@latest",
        },
        "katana": {
            "go": "github.com/projectdiscovery/katana/cmd/katana@latest",
        },
        "hakrawler": {
            "go": "github.com/hakluke/hakrawler@latest",
        },
        "gospider": {
            "go": "github.com/jaeles-project/gospider@latest",
        },
        "sslscan": {
            SystemType.KALI: "sslscan",
            SystemType.UBUNTU: "sslscan",
        },
        "testssl": {
            SystemType.KALI: "testssl.sh",
            "git": "https://github.com/drwetter/testssl.sh.git",
        },
        "dnsrecon": {
            SystemType.KALI: "dnsrecon",
            "pip": "dnsrecon",
        },
        "dnsenum": {
            SystemType.KALI: "dnsenum",
        },
        "fierce": {
            SystemType.KALI: "fierce",
            "pip": "fierce",
        },
        "massdns": {
            "git": "https://github.com/blechschmidt/massdns.git",
        },
        "trivy": {
            "apt": "trivy",
            "brew": "trivy",
        },
        "prowler": {
            "pip": "prowler",
        },
        "scoutsuite": {
            "pip": "scoutsuite",
        },
        "pacu": {
            "pip": "pacu",
        },
        # Add Ollama
        "ollama": {
            SystemType.KALI: "curl -fsSL https://ollama.com/install.sh | sh",
            SystemType.UBUNTU: "curl -fsSL https://ollama.com/install.sh | sh",
            SystemType.DEBIAN: "curl -fsSL https://ollama.com/install.sh | sh",
            SystemType.ARCH: "yay -S ollama",
            SystemType.MACOS: "brew install ollama",
            SystemType.TERMUX: "pkg install ollama",
            "script": "curl -fsSL https://ollama.com/install.sh | sh",
        },
    }
    
    # Essential tools per category
    ESSENTIAL_TOOLS = {
        "core": ["nmap", "curl", "wget", "git", "python3", "pip"],
        "ai": ["ollama"],
        "recon": ["nmap", "masscan", "subfinder", "amass", "theHarvester"],
        "web": ["nikto", "sqlmap", "gobuster", "ffuf", "nuclei", "wpscan"],
        "exploit": ["hydra", "john", "hashcat", "metasploit"],
        "forensics": ["volatility", "foremost", "binwalk", "exiftool"],
        "network": ["wireshark", "tcpdump", "netcat"],
        "defense": ["lynis", "rkhunter", "clamav", "fail2ban"],
    }
    
    def __init__(self, system_info: Dict, ai: OllamaAI = None):
        self.system = system_info
        self.system_type = system_info.get("type", SystemType.UNKNOWN)
        self.install_cmd = system_info.get("install_cmd", "apt install -y")
        self.ai = ai
        self.installed = []
        self.failed = []
    
    def check_tool(self, tool: str) -> bool:
        """Check if tool is installed"""
        return shutil.which(tool) is not None
    
    def get_missing_tools(self, tools: List[str]) -> List[str]:
        """Get list of missing tools"""
        return [t for t in tools if not self.check_tool(t)]
    
    def get_package_name(self, tool: str) -> Optional[str]:
        """Get package name for tool on current system"""
        if tool not in self.TOOL_PACKAGES:
            return tool  # Try tool name directly
        
        packages = self.TOOL_PACKAGES[tool]
        
        # Check system-specific first
        if self.system_type in packages:
            return packages[self.system_type]
        
        # Check pip
        if "pip" in packages and self.check_tool("pip3"):
            return ("pip", packages["pip"])
        
        # Check go
        if "go" in packages and self.check_tool("go"):
            return ("go", packages["go"])
        
        # Check git
        if "git" in packages:
            return ("git", packages["git"])
        
        # Check script
        if "script" in packages:
            return ("script", packages["script"])
        
        return tool  # Fallback to tool name
    
    def install_tool(self, tool: str, verbose: bool = True) -> bool:
        """Install a single tool"""
        if self.check_tool(tool):
            if verbose:
                print(f"    {C.GRN}[✓]{C.R} {tool} already installed")
            return True
        
        package = self.get_package_name(tool)
        
        if verbose:
            print(f"    {C.YEL}[→]{C.R} Installing {tool}...")
        
        try:
            if isinstance(package, tuple):
                method, pkg = package
                
                if method == "pip":
                    cmd = f"pip3 install --break-system-packages {pkg}"
                elif method == "go":
                    cmd = f"go install {pkg}"
                elif method == "git":
                    clone_dir = f"/opt/{tool}"
                    cmd = f"git clone {pkg} {clone_dir}"
                elif method == "script":
                    cmd = pkg
                else:
                    cmd = f"{self.install_cmd} {pkg}"
            else:
                cmd = f"{self.install_cmd} {package}"
            
            # Execute installation
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=300
            )
            
            if result.returncode == 0 or self.check_tool(tool):
                self.installed.append(tool)
                if verbose:
                    print(f"    {C.GRN}[✓]{C.R} {tool} installed successfully")
                return True
            else:
                self.failed.append((tool, result.stderr[:200]))
                if verbose:
                    print(f"    {C.RED}[✗]{C.R} {tool} installation failed")
                    if self.ai and self.ai.is_available:
                        fix = self.ai.fix_error(tool, result.stderr[:500])
                        print(f"    {C.CYN}[AI]{C.R} {fix[:200]}")
                return False
                
        except Exception as e:
            self.failed.append((tool, str(e)))
            if verbose:
                print(f"    {C.RED}[✗]{C.R} {tool}: {str(e)[:100]}")
            return False
    
    def install_tools(self, tools: List[str], parallel: bool = False) -> Dict:
        """Install multiple tools"""
        missing = self.get_missing_tools(tools)
        
        if not missing:
            print(f"    {C.GRN}[✓]{C.R} All {len(tools)} tools already installed!")
            return {"installed": [], "failed": [], "skipped": tools}
        
        print(f"\n    {C.CYN}[*]{C.R} Installing {len(missing)} missing tools...")
        
        if parallel:
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(self.install_tool, t, False): t for t in missing}
                for future in as_completed(futures):
                    tool = futures[future]
                    try:
                        success = future.result()
                        status = f"{C.GRN}✓{C.R}" if success else f"{C.RED}✗{C.R}"
                        print(f"    [{status}] {tool}")
                    except Exception as e:
                        print(f"    [{C.RED}✗{C.R}] {tool}: {e}")
        else:
            for tool in missing:
                self.install_tool(tool)
        
        return {
            "installed": self.installed,
            "failed": self.failed,
            "skipped": [t for t in tools if t not in missing]
        }
    
    def install_category(self, category: str) -> Dict:
        """Install all tools in a category"""
        if category not in self.ESSENTIAL_TOOLS:
            print(f"    {C.RED}[!]{C.R} Unknown category: {category}")
            return {}
        
        tools = self.ESSENTIAL_TOOLS[category]
        print(f"\n    {C.CYN}[*]{C.R} Installing {category.upper()} tools ({len(tools)} tools)...")
        return self.install_tools(tools)
    
    def install_all_essential(self) -> Dict:
        """Install all essential tools"""
        all_tools = []
        for tools in self.ESSENTIAL_TOOLS.values():
            all_tools.extend(tools)
        
        all_tools = list(set(all_tools))  # Remove duplicates
        print(f"\n    {C.CYN}[*]{C.R} Installing ALL essential tools ({len(all_tools)} tools)...")
        return self.install_tools(all_tools)
    
    def install_ollama(self) -> bool:
        """Install Ollama AI"""
        if self.check_tool("ollama"):
            print(f"    {C.GRN}[✓]{C.R} Ollama already installed")
            return True
        
        print(f"\n    {C.CYN}[AI]{C.R} Installing Ollama...")
        
        try:
            if self.system_type == SystemType.MACOS:
                cmd = "brew install ollama"
            elif self.system_type == SystemType.ARCH:
                cmd = "yay -S --noconfirm ollama-bin"
            else:
                cmd = "curl -fsSL https://ollama.com/install.sh | sh"
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0 or self.check_tool("ollama"):
                print(f"    {C.GRN}[✓]{C.R} Ollama installed successfully")
                return True
            else:
                print(f"    {C.RED}[✗]{C.R} Ollama installation failed")
                return False
        except Exception as e:
            print(f"    {C.RED}[✗]{C.R} Error: {e}")
            return False
    
    def install_ai_models(self, models: List[str] = None) -> Dict:
        """Install AI models"""
        if not self.check_tool("ollama"):
            print(f"    {C.RED}[!]{C.R} Ollama not installed. Installing first...")
            self.install_ollama()
        
        if not self.check_tool("ollama"):
            return {"error": "Failed to install Ollama"}
        
        # Start Ollama service
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        
        default_models = [
            "deepseek-r1:7b",      # Best reasoning (smaller version)
            "llama3.2:3b",          # Fast general purpose
            "codestral:latest",    # Best for code
        ]
        
        models = models or default_models
        results = {"installed": [], "failed": []}
        
        for model in models:
            print(f"\n    {C.CYN}[AI]{C.R} Pulling {model}... (this may take a while)")
            try:
                result = subprocess.run(
                    ["ollama", "pull", model],
                    capture_output=True, text=True, timeout=3600
                )
                if result.returncode == 0:
                    results["installed"].append(model)
                    print(f"    {C.GRN}[✓]{C.R} {model} installed")
                else:
                    results["failed"].append(model)
                    print(f"    {C.RED}[✗]{C.R} {model} failed")
            except Exception as e:
                results["failed"].append(model)
                print(f"    {C.RED}[✗]{C.R} {model}: {e}")
        
        return results
    
    def full_setup(self) -> Dict:
        """Complete system setup with AI and all tools"""
        results = {
            "system": self.system,
            "ollama": False,
            "ai_models": [],
            "tools": {}
        }
        
        # 1. System update
        print(f"\n{C.CYN}{'═' * 60}")
        print(f"    PHASE 1: System Update")
        print(f"{'═' * 60}{C.R}")
        
        if self.system_type == SystemType.TERMUX:
            subprocess.run("pkg update -y && pkg upgrade -y", shell=True)
        elif self.system_type in [SystemType.KALI, SystemType.UBUNTU, SystemType.DEBIAN]:
            subprocess.run("apt update && apt upgrade -y", shell=True)
        elif self.system_type == SystemType.ARCH:
            subprocess.run("pacman -Syu --noconfirm", shell=True)
        
        # 2. Install Ollama
        print(f"\n{C.CYN}{'═' * 60}")
        print(f"    PHASE 2: AI Setup (Ollama)")
        print(f"{'═' * 60}{C.R}")
        
        results["ollama"] = self.install_ollama()
        
        # 3. Install AI Models
        print(f"\n{C.CYN}{'═' * 60}")
        print(f"    PHASE 3: AI Models Download")
        print(f"{'═' * 60}{C.R}")
        
        ai_results = self.install_ai_models()
        results["ai_models"] = ai_results.get("installed", [])
        
        # 4. Install All Tools
        print(f"\n{C.CYN}{'═' * 60}")
        print(f"    PHASE 4: Security Tools Installation")
        print(f"{'═' * 60}{C.R}")
        
        results["tools"] = self.install_all_essential()
        
        return results

# ═══════════════════════════════════════════════════════════════════════════════
# COLORS & LOGGING
# ═══════════════════════════════════════════════════════════════════════════════
class C:
    R = '\033[0m'; B = '\033[1m'; D = '\033[2m'; I = '\033[3m'; U = '\033[4m'
    RED = '\033[91m'; GRN = '\033[92m'; YEL = '\033[93m'; BLU = '\033[94m'
    MAG = '\033[95m'; CYN = '\033[96m'; WHT = '\033[97m'; GRY = '\033[90m'
    ORG = '\033[38;5;208m'; PNK = '\033[38;5;206m'; LME = '\033[38;5;118m'
    BG_RED = '\033[41m'; BG_GRN = '\033[42m'; BG_BLU = '\033[44m'
    BG_YEL = '\033[43m'; BG_MAG = '\033[45m'; BG_CYN = '\033[46m'; BG_BLK = '\033[40m'

class Log:
    ai: OllamaAI = None
    
    @staticmethod
    def info(m): print(f"    {C.CYN}[*]{C.R} {m}")
    @staticmethod
    def ok(m): print(f"    {C.GRN}[✓]{C.R} {m}")
    @staticmethod
    def err(m): print(f"    {C.RED}[✗]{C.R} {m}")
    @staticmethod
    def warn(m): print(f"    {C.YEL}[!]{C.R} {m}")
    @staticmethod
    def ai_msg(m): print(f"    {C.MAG}[🤖]{C.R} {m}")
    @staticmethod
    def tool(name, status="running"):
        icons = {"running": "⚡", "done": "✓", "error": "✗", "skip": "○", "ai": "🤖"}
        colors = {"running": C.YEL, "done": C.GRN, "error": C.RED, "skip": C.GRY, "ai": C.MAG}
        print(f"    {colors.get(status, C.WHT)}[{icons.get(status, '•')}]{C.R} {name}")
    @staticmethod
    def section(title, icon=""):
        print(f"\n{C.CYN}{'═' * 100}")
        print(f"║ {icon} {C.B}{title}{C.R}{C.CYN}")
        print(f"{'═' * 100}{C.R}")
    @staticmethod
    def ai_think(prompt: str):
        """Show AI thinking animation"""
        if Log.ai and Log.ai.is_available:
            print(f"    {C.MAG}[🤖]{C.R} AI analyzing...", end='', flush=True)
            response = Log.ai.chat(prompt)
            print(f"\r    {C.MAG}[🤖]{C.R} {response[:150]}...")
            return response
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
class Config:
    OUTPUT_DIR = "ayed_v15_results"
    TIMEOUT = 600
    THREADS = 25
    VERBOSE = True
    AI_ENABLED = True
    AI_ASSIST_LEVEL = "full"  # minimal, moderate, full

# ═══════════════════════════════════════════════════════════════════════════════
# AI-POWERED TOOL EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════
class AIToolExecutor:
    """Tool Executor with AI Integration"""
    
    def __init__(self, output_dir: str, ai: OllamaAI = None, installer: SmartInstaller = None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}
        self.ai = ai
        self.installer = installer
        self.lock = threading.Lock()
    
    def check_tool(self, tool: str) -> bool:
        return shutil.which(tool) is not None
    
    def auto_install(self, tool: str) -> bool:
        """Auto-install missing tool"""
        if self.installer:
            return self.installer.install_tool(tool)
        return False
    
    def run(self, name: str, cmd: List[str], timeout: int = Config.TIMEOUT,
            shell: bool = False, auto_install: bool = True, ai_analyze: bool = True) -> Dict:
        """Execute tool with AI analysis"""
        
        result = {
            "tool": name,
            "command": " ".join(cmd) if isinstance(cmd, list) else cmd,
            "status": "pending",
            "output": "",
            "error": "",
            "duration": 0,
            "ai_analysis": None,
            "timestamp": datetime.now().isoformat()
        }
        
        tool_name = cmd[0] if isinstance(cmd, list) else cmd.split()[0]
        
        # Check if tool exists, auto-install if needed
        if not shell and not self.check_tool(tool_name):
            if auto_install and self.installer:
                Log.warn(f"{tool_name} not found, attempting auto-install...")
                if self.auto_install(tool_name):
                    Log.ok(f"{tool_name} installed successfully")
                else:
                    result["status"] = "not_installed"
                    result["error"] = f"{tool_name} not found and could not be installed"
                    
                    # AI suggestion for alternatives
                    if self.ai and self.ai.is_available:
                        suggestion = self.ai.chat(
                            f"Tool '{tool_name}' is not available. Suggest 3 alternative tools that do the same thing.",
                            system_prompt="You are a security tools expert. Give brief, practical suggestions."
                        )
                        result["ai_analysis"] = f"Alternatives: {suggestion}"
                        Log.ai_msg(f"Alternatives: {suggestion[:100]}...")
                    
                    Log.tool(name, "skip")
                    return result
            else:
                result["status"] = "not_installed"
                result["error"] = f"{tool_name} not found"
                Log.tool(name, "skip")
                return result
        
        Log.tool(name, "running")
        start = time.time()
        
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, shell=shell
            )
            stdout, stderr = proc.communicate(timeout=timeout)
            result["output"] = stdout
            result["error"] = stderr
            result["status"] = "success" if proc.returncode == 0 else "failed"
            result["return_code"] = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            result["status"] = "timeout"
            result["error"] = f"Timeout after {timeout}s"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        
        result["duration"] = time.time() - start
        
        # Save output to file
        if result["output"]:
            output_file = f"{self.output_dir}/{name.replace(' ', '_')}_{int(time.time())}.txt"
            with open(output_file, 'w') as f:
                f.write(f"Command: {result['command']}\n{'='*80}\n{result['output']}")
            result["output_file"] = output_file
        
        # AI Analysis
        if ai_analyze and self.ai and self.ai.is_available and result["output"]:
            try:
                analysis = self.ai.analyze_results(name, result["output"][:2000])
                result["ai_analysis"] = analysis
                # Show brief AI insight
                if analysis and len(analysis) > 20:
                    Log.ai_msg(f"Analysis: {analysis[:100]}...")
            except:
                pass
        
        Log.tool(name, "done" if result["status"] == "success" else "error")
        
        with self.lock:
            self.results[name] = result
        
        return result
    
    def run_parallel(self, tasks: List[Tuple], timeout: int = Config.TIMEOUT) -> List[Dict]:
        """Run multiple tools in parallel"""
        results = []
        with ThreadPoolExecutor(max_workers=Config.THREADS) as executor:
            futures = {executor.submit(self.run, name, cmd, timeout): name for name, cmd in tasks}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    Log.err(f"{futures[future]}: {e}")
        return results

# ═══════════════════════════════════════════════════════════════════════════════
# AI-ENHANCED SECURITY TEAMS
# ═══════════════════════════════════════════════════════════════════════════════
class AIRedTeam:
    """AI-Enhanced Red Team Operations"""
    
    def __init__(self, target: str, executor: AIToolExecutor, ai: OllamaAI = None):
        self.target = target
        self.domain = urlparse(target).netloc if target.startswith('http') else target
        self.ip = self._resolve_ip()
        self.executor = executor
        self.ai = ai
        self.out = executor.output_dir
    
    def _resolve_ip(self) -> str:
        try: return socket.gethostbyname(self.domain.split(':')[0])
        except: return self.domain
    
    def ai_recon_plan(self) -> str:
        """AI creates reconnaissance plan"""
        if self.ai and self.ai.is_available:
            return self.ai.chat(
                f"Create a reconnaissance plan for target: {self.target}",
                system_prompt="""You are a penetration testing expert. Create a brief, 
                actionable reconnaissance plan including:
                1. Information gathering steps
                2. Tools to use in order
                3. What to look for
                Be concise and tactical."""
            )
        return "AI not available - using default reconnaissance methodology"
    
    # SCANNING TOOLS
    def nmap_full(self):
        return self.executor.run("Nmap Full Scan", [
            "nmap", "-sS", "-sV", "-sC", "-O", "-A", "-p-", "-T4",
            "--script=default,safe,vuln",
            "-oA", f"{self.out}/nmap_full", self.ip
        ])
    
    def nmap_vuln(self):
        return self.executor.run("Nmap Vulnerability Scan", [
            "nmap", "-sV", "--script=vuln,exploit",
            "-p", "21,22,23,25,53,80,443,445,3306,3389,8080",
            "-oX", f"{self.out}/nmap_vuln.xml", self.ip
        ])
    
    def masscan_full(self):
        return self.executor.run("Masscan Full Ports", [
            "masscan", self.ip, "-p", "0-65535",
            "--rate=10000", "-oJ", f"{self.out}/masscan.json"
        ])
    
    def rustscan(self):
        return self.executor.run("RustScan Fast", [
            "rustscan", "-a", self.ip, "--ulimit", "5000",
            "--", "-sV", "-sC"
        ])
    
    # WEB SCANNING
    def nikto_scan(self):
        return self.executor.run("Nikto Web Scan", [
            "nikto", "-h", self.target, "-C", "all",
            "-Format", "html", "-o", f"{self.out}/nikto.html"
        ])
    
    def nuclei_scan(self):
        return self.executor.run("Nuclei Vulnerability Scan", [
            "nuclei", "-u", self.target,
            "-severity", "critical,high,medium",
            "-o", f"{self.out}/nuclei.txt"
        ])
    
    def wpscan_full(self):
        return self.executor.run("WPScan WordPress", [
            "wpscan", "--url", self.target,
            "--enumerate", "vp,vt,u",
            "-o", f"{self.out}/wpscan.json", "-f", "json"
        ])
    
    def whatweb_scan(self):
        return self.executor.run("WhatWeb Fingerprint", [
            "whatweb", self.target, "-a", "3", "-v",
            "--log-json", f"{self.out}/whatweb.json"
        ])
    
    def wafw00f_scan(self):
        return self.executor.run("WAF Detection", [
            "wafw00f", self.target, "-a", "-o", f"{self.out}/waf.txt"
        ])
    
    # SQL INJECTION
    def sqlmap_scan(self):
        return self.executor.run("SQLMap Injection Test", [
            "sqlmap", "-u", self.target, "--batch",
            "--level=3", "--risk=2", "--threads=5",
            "--output-dir", f"{self.out}/sqlmap"
        ])
    
    # XSS
    def xsstrike_scan(self):
        return self.executor.run("XSStrike XSS Test", [
            "xsstrike", "-u", self.target, "--crawl", "-t", "10"
        ])
    
    def dalfox_scan(self):
        return self.executor.run("Dalfox XSS Scanner", [
            "dalfox", "url", self.target,
            "-o", f"{self.out}/dalfox.txt"
        ])
    
    # DIRECTORY DISCOVERY
    def gobuster_dir(self):
        return self.executor.run("Gobuster Directory Scan", [
            "gobuster", "dir", "-u", self.target,
            "-w", "/usr/share/wordlists/dirb/common.txt",
            "-x", "php,html,txt,bak",
            "-t", "50", "-o", f"{self.out}/gobuster.txt"
        ])
    
    def ffuf_scan(self):
        return self.executor.run("FFUF Fuzzing", [
            "ffuf", "-u", f"{self.target}/FUZZ",
            "-w", "/usr/share/wordlists/dirb/common.txt",
            "-mc", "200,301,302,401,403",
            "-o", f"{self.out}/ffuf.json", "-of", "json"
        ])
    
    def feroxbuster_scan(self):
        return self.executor.run("Feroxbuster Recursive", [
            "feroxbuster", "-u", self.target,
            "-x", "php,html,txt", "-t", "50",
            "-o", f"{self.out}/feroxbuster.txt"
        ])
    
    # SUBDOMAIN DISCOVERY
    def subfinder_scan(self):
        return self.executor.run("Subfinder Subdomains", [
            "subfinder", "-d", self.domain, "-all",
            "-o", f"{self.out}/subdomains.txt"
        ])
    
    def amass_enum(self):
        return self.executor.run("Amass Enumeration", [
            "amass", "enum", "-d", self.domain,
            "-o", f"{self.out}/amass.txt"
        ])
    
    # SSL ANALYSIS
    def sslscan_full(self):
        return self.executor.run("SSLScan Analysis", [
            "sslscan", "--show-certificate", self.domain
        ])
    
    def testssl_scan(self):
        return self.executor.run("TestSSL Analysis", [
            "testssl.sh", "--full", self.domain
        ])
    
    # DNS
    def dnsrecon_scan(self):
        return self.executor.run("DNSRecon Enumeration", [
            "dnsrecon", "-d", self.domain, "-t", "std,brt",
            "-j", f"{self.out}/dnsrecon.json"
        ])
    
    def run_all(self, ai_plan: bool = True) -> List[Dict]:
        """Run all Red Team tools with AI guidance"""
        Log.section("🔴 RED TEAM - AI-ENHANCED OFFENSIVE OPERATIONS", "🎯")
        
        # AI creates attack plan
        if ai_plan and self.ai and self.ai.is_available:
            plan = self.ai_recon_plan()
            print(f"\n    {C.MAG}[🤖 AI ATTACK PLAN]{C.R}")
            print(f"    {'-' * 60}")
            for line in plan.split('\n')[:10]:
                print(f"    {C.GRY}{line}{C.R}")
            print(f"    {'-' * 60}\n")
        
        tools = [
            self.nmap_full, self.nmap_vuln, self.nikto_scan,
            self.nuclei_scan, self.whatweb_scan, self.wafw00f_scan,
            self.sqlmap_scan, self.gobuster_dir, self.subfinder_scan,
            self.sslscan_full, self.dnsrecon_scan
        ]
        
        results = []
        for func in tools:
            try:
                results.append(func())
            except Exception as e:
                Log.err(f"{func.__name__}: {e}")
        
        # AI Summary
        if self.ai and self.ai.is_available:
            summary = self.ai.suggest_next_step(results)
            print(f"\n    {C.MAG}[🤖 AI RECOMMENDATIONS]{C.R}")
            print(f"    {summary[:300]}...")
        
        return results


class AIBlueTeam:
    """AI-Enhanced Blue Team Operations"""
    
    def __init__(self, target: str, executor: AIToolExecutor, ai: OllamaAI = None):
        self.target = target
        self.executor = executor
        self.ai = ai
        self.out = executor.output_dir
    
    def ai_defense_plan(self) -> str:
        """AI creates defense assessment plan"""
        if self.ai and self.ai.is_available:
            return self.ai.chat(
                f"Create a security hardening assessment plan",
                system_prompt="""You are a defensive security expert. Create a brief 
                security audit plan including:
                1. System hardening checks
                2. Malware detection steps
                3. Log analysis priorities
                Be concise and thorough."""
            )
        return "AI not available - using default defensive methodology"
    
    def lynis_audit(self):
        return self.executor.run("Lynis System Audit", [
            "lynis", "audit", "system", "--quick",
            "--report-file", f"{self.out}/lynis.dat"
        ])
    
    def rkhunter_scan(self):
        return self.executor.run("RKHunter Rootkit Scan", [
            "rkhunter", "--check", "--skip-keypress",
            "--logfile", f"{self.out}/rkhunter.log"
        ])
    
    def chkrootkit_scan(self):
        return self.executor.run("Chkrootkit Scan", ["chkrootkit", "-q"])
    
    def clamav_scan(self):
        return self.executor.run("ClamAV Malware Scan", [
            "clamscan", "-r", "--infected", "/home",
            "--log", f"{self.out}/clamav.log"
        ], timeout=1800)
    
    def fail2ban_status(self):
        return self.executor.run("Fail2ban Status", [
            "fail2ban-client", "status"
        ])
    
    def security_headers(self):
        return self.executor.run("Security Headers Check",
            f"curl -sI {self.target} | grep -iE '(x-frame|x-content|strict-transport|content-security)'",
            shell=True)
    
    def ssl_check(self):
        domain = urlparse(self.target).netloc if self.target.startswith('http') else self.target
        return self.executor.run("SSL Certificate Check",
            f"echo | openssl s_client -connect {domain}:443 2>/dev/null | openssl x509 -noout -dates",
            shell=True)
    
    def firewall_status(self):
        return self.executor.run("Firewall Status",
            "ufw status verbose || iptables -L -n",
            shell=True)
    
    def open_ports(self):
        return self.executor.run("Open Ports Check",
            "ss -tulpn | grep LISTEN",
            shell=True)
    
    def running_services(self):
        return self.executor.run("Running Services",
            "systemctl list-units --type=service --state=running",
            shell=True)
    
    def run_all(self, ai_plan: bool = True) -> List[Dict]:
        """Run all Blue Team tools with AI guidance"""
        Log.section("🔵 BLUE TEAM - AI-ENHANCED DEFENSIVE OPERATIONS", "🛡️")
        
        if ai_plan and self.ai and self.ai.is_available:
            plan = self.ai_defense_plan()
            print(f"\n    {C.MAG}[🤖 AI DEFENSE PLAN]{C.R}")
            print(f"    {'-' * 60}")
            for line in plan.split('\n')[:10]:
                print(f"    {C.GRY}{line}{C.R}")
            print(f"    {'-' * 60}\n")
        
        tools = [
            self.lynis_audit, self.rkhunter_scan, self.chkrootkit_scan,
            self.fail2ban_status, self.security_headers, self.ssl_check,
            self.firewall_status, self.open_ports, self.running_services
        ]
        
        results = []
        for func in tools:
            try:
                results.append(func())
            except Exception as e:
                Log.err(f"{func.__name__}: {e}")
        
        return results


class AIBlackTeam:
    """AI-Enhanced Black Team Operations"""
    
    def __init__(self, target: str, executor: AIToolExecutor, ai: OllamaAI = None):
        self.target = target
        self.domain = urlparse(target).netloc if target.startswith('http') else target
        self.ip = socket.gethostbyname(self.domain.split(':')[0]) if self.domain else target
        self.executor = executor
        self.ai = ai
        self.out = executor.output_dir
    
    def hydra_ssh(self):
        return self.executor.run("Hydra SSH Bruteforce", [
            "hydra", "-L", "/usr/share/wordlists/metasploit/unix_users.txt",
            "-P", "/usr/share/wordlists/rockyou.txt",
            "-t", "4", "-f", f"ssh://{self.ip}"
        ], timeout=300)
    
    def hydra_ftp(self):
        return self.executor.run("Hydra FTP Bruteforce", [
            "hydra", "-L", "/usr/share/wordlists/metasploit/unix_users.txt",
            "-P", "/usr/share/wordlists/rockyou.txt",
            "-t", "4", "-f", f"ftp://{self.ip}"
        ], timeout=300)
    
    def enum4linux_scan(self):
        return self.executor.run("Enum4linux SMB Enum", [
            "enum4linux", "-a", self.ip
        ])
    
    def smbclient_list(self):
        return self.executor.run("SMBclient List Shares", [
            "smbclient", "-L", f"//{self.ip}", "-N"
        ])
    
    def smbmap_scan(self):
        return self.executor.run("SMBmap Scan", [
            "smbmap", "-H", self.ip
        ])
    
    def crackmapexec_smb(self):
        return self.executor.run("CrackMapExec SMB", [
            "crackmapexec", "smb", self.ip
        ])
    
    def responder_analyze(self):
        return self.executor.run("Responder Analysis Mode", [
            "responder", "-I", "eth0", "-A"
        ], timeout=30)
    
    def msfvenom_linux(self):
        return self.executor.run("MSFvenom Linux Payload", [
            "msfvenom", "-p", "linux/x64/shell_reverse_tcp",
            "LHOST=127.0.0.1", "LPORT=4444",
            "-f", "elf", "-o", f"{self.out}/shell.elf"
        ])
    
    def run_all(self) -> List[Dict]:
        """Run all Black Team tools"""
        Log.section("⚫ BLACK TEAM - AI-ENHANCED EXPLOITATION", "💀")
        
        tools = [
            self.enum4linux_scan, self.smbclient_list, self.smbmap_scan,
            self.crackmapexec_smb
        ]
        
        results = []
        for func in tools:
            try:
                results.append(func())
            except Exception as e:
                Log.err(f"{func.__name__}: {e}")
        
        return results


class AIForensics:
    """AI-Enhanced Digital Forensics"""
    
    def __init__(self, evidence_path: str, executor: AIToolExecutor, ai: OllamaAI = None):
        self.evidence = evidence_path
        self.executor = executor
        self.ai = ai
        self.out = executor.output_dir
    
    def ai_forensics_plan(self) -> str:
        """AI creates forensics investigation plan"""
        if self.ai and self.ai.is_available:
            return self.ai.chat(
                "Create a digital forensics investigation plan",
                system_prompt="""You are a digital forensics expert. Create a brief 
                investigation plan including:
                1. Evidence acquisition steps
                2. Analysis priorities
                3. Timeline reconstruction approach
                Be methodical and legally sound."""
            )
        return "AI not available - using standard forensics methodology"
    
    def strings_extract(self):
        return self.executor.run("Strings Extraction",
            f"strings -a {self.evidence}/* 2>/dev/null | head -1000",
            shell=True)
    
    def exiftool_meta(self):
        return self.executor.run("ExifTool Metadata", [
            "exiftool", "-r", "-json", self.evidence
        ])
    
    def binwalk_analyze(self):
        return self.executor.run("Binwalk Analysis", [
            "binwalk", "-e", "-M", self.evidence
        ])
    
    def foremost_carve(self):
        return self.executor.run("Foremost File Carving", [
            "foremost", "-i", self.evidence, "-o", f"{self.out}/foremost"
        ])
    
    def volatility_info(self):
        return self.executor.run("Volatility Image Info", [
            "vol", "-f", self.evidence, "windows.info"
        ])
    
    def hash_files(self):
        return self.executor.run("Hash Evidence Files",
            f"find {self.evidence} -type f -exec md5sum {{}} \\; > {self.out}/hashes.md5",
            shell=True)
    
    def run_all(self) -> List[Dict]:
        """Run all Forensics tools"""
        Log.section("🟢 FORENSICS - AI-ENHANCED INVESTIGATION", "🔬")
        
        if self.ai and self.ai.is_available:
            plan = self.ai_forensics_plan()
            print(f"\n    {C.MAG}[🤖 AI FORENSICS PLAN]{C.R}")
            for line in plan.split('\n')[:8]:
                print(f"    {C.GRY}{line}{C.R}")
            print()
        
        tools = [
            self.strings_extract, self.exiftool_meta, self.hash_files
        ]
        
        results = []
        for func in tools:
            try:
                results.append(func())
            except Exception as e:
                Log.err(f"{func.__name__}: {e}")
        
        return results


class AIOSINT:
    """AI-Enhanced OSINT Operations"""
    
    def __init__(self, target: str, executor: AIToolExecutor, ai: OllamaAI = None):
        self.target = target
        self.domain = urlparse(target).netloc if target.startswith('http') else target
        self.executor = executor
        self.ai = ai
        self.out = executor.output_dir
    
    def theharvester_scan(self):
        return self.executor.run("TheHarvester OSINT", [
            "theHarvester", "-d", self.domain, "-b", "all",
            "-f", f"{self.out}/harvester"
        ])
    
    def whois_lookup(self):
        return self.executor.run("WHOIS Lookup", [
            "whois", self.domain
        ])
    
    def dig_all(self):
        return self.executor.run("DIG DNS Records", [
            "dig", self.domain, "ANY", "+noall", "+answer"
        ])
    
    def shodan_host(self):
        return self.executor.run("Shodan Host Info", [
            "shodan", "host", self.domain
        ])
    
    def crtsh_lookup(self):
        return self.executor.run("crt.sh Certificate Search",
            f"curl -s 'https://crt.sh/?q=%.{self.domain}&output=json' | jq '.[].name_value' | sort -u",
            shell=True)
    
    def wayback_urls(self):
        return self.executor.run("Wayback URLs", [
            "waybackurls", self.domain
        ])
    
    def run_all(self) -> List[Dict]:
        """Run all OSINT tools"""
        Log.section("⚪ OSINT - AI-ENHANCED INTELLIGENCE", "🔍")
        
        tools = [
            self.theharvester_scan, self.whois_lookup, self.dig_all,
            self.crtsh_lookup, self.wayback_urls
        ]
        
        results = []
        for func in tools:
            try:
                results.append(func())
            except Exception as e:
                Log.err(f"{func.__name__}: {e}")
        
        return results

# ═══════════════════════════════════════════════════════════════════════════════
# AI-POWERED REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
class AIReportGenerator:
    """AI-Enhanced Report Generation"""
    
    @staticmethod
    def generate(target: str, results: List[Dict], output_dir: str, ai: OllamaAI = None) -> Tuple[str, str, str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Statistics
        total = len(results)
        success = len([r for r in results if r.get("status") == "success"])
        failed = len([r for r in results if r.get("status") in ["failed", "error"]])
        missing = len([r for r in results if r.get("status") == "not_installed"])
        
        # AI Executive Summary
        ai_summary = ""
        if ai and ai.is_available:
            ai_summary = ai.generate_report_summary(results)
        
        # JSON Report
        json_file = f"{output_dir}/report_{timestamp}.json"
        report_data = {
            "scan_info": {
                "version": VERSION,
                "timestamp": datetime.now().isoformat(),
                "target": target,
                "ai_enabled": ai is not None and ai.is_available
            },
            "summary": {
                "total_tools": total,
                "successful": success,
                "failed": failed,
                "not_installed": missing,
            },
            "ai_executive_summary": ai_summary,
            "results": results
        }
        
        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # HTML Report with AI insights
        html_file = f"{output_dir}/report_{timestamp}.html"
        
        tools_html = ""
        for r in results:
            status_color = {
                "success": "#28a745", "failed": "#dc3545",
                "error": "#dc3545", "timeout": "#ffc107",
                "not_installed": "#6c757d"
            }.get(r.get("status", ""), "#6c757d")
            
            ai_insight = r.get("ai_analysis", "")
            ai_section = f"""
                <div style="background:#2a1a4a;padding:10px;border-radius:4px;margin-top:10px;">
                    <span style="color:#a855f7;">🤖 AI Analysis:</span>
                    <p style="color:#888;font-size:12px;margin:5px 0 0 0;">{ai_insight[:300] if ai_insight else 'No AI analysis'}</p>
                </div>
            """ if ai_insight else ""
            
            tools_html += f"""
            <div style="background:#1a1a2e;border-radius:8px;padding:15px;margin:10px 0;border-left:4px solid {status_color};">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <h4 style="color:#fff;margin:0;">{r.get('tool', 'Unknown')}</h4>
                    <span style="background:{status_color};color:#fff;padding:3px 10px;border-radius:3px;font-size:12px;">{r.get('status', 'N/A')}</span>
                </div>
                <p style="color:#666;font-size:11px;margin:5px 0;">Duration: {r.get('duration', 0):.1f}s</p>
                {ai_section}
            </div>"""
        
        html_content = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Ayed v{VERSION} AI Report</title>
<style>
body{{background:#0a0a14;color:#fff;font-family:'Segoe UI',sans-serif;padding:20px}}
.container{{max-width:1200px;margin:0 auto}}
.header{{text-align:center;padding:30px;background:linear-gradient(135deg,#1a1a2e,#2a1a4a);border-radius:10px;margin-bottom:20px}}
.header h1{{color:#a855f7;margin:0}}
.ai-box{{background:#2a1a4a;border:1px solid #a855f7;border-radius:10px;padding:20px;margin:20px 0}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:20px 0}}
.stat{{background:#1a1a2e;padding:20px;border-radius:8px;text-align:center}}
.stat h2{{margin:0;font-size:2em}}
</style></head><body>
<div class="container">
<div class="header">
    <h1>🤖 Ayed v{VERSION}</h1>
    <p style="color:#888;">AI-Powered Security Assessment Report</p>
    <p style="color:#666;">Target: {target} | {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
</div>

<div class="ai-box">
    <h3 style="color:#a855f7;margin-top:0;">🤖 AI Executive Summary</h3>
    <p style="color:#ccc;">{ai_summary if ai_summary else 'AI analysis not available'}</p>
</div>

<div class="stats">
    <div class="stat"><h2 style="color:#00d4ff;">{total}</h2><p>Total Tools</p></div>
    <div class="stat"><h2 style="color:#28a745;">{success}</h2><p>Successful</p></div>
    <div class="stat"><h2 style="color:#dc3545;">{failed}</h2><p>Failed</p></div>
    <div class="stat"><h2 style="color:#6c757d;">{missing}</h2><p>Missing</p></div>
</div>

<h3 style="border-bottom:2px solid #a855f7;padding-bottom:10px;">Tool Results</h3>
{tools_html}

<div style="text-align:center;padding:20px;color:#666;font-size:12px;">
    <p>Generated by Ayed v{VERSION} with AI Enhancement</p>
</div>
</div></body></html>"""
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        # Markdown Report
        md_file = f"{output_dir}/report_{timestamp}.md"
        md_content = f"""# 🤖 Ayed v{VERSION} AI Security Report

## Target: {target}
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

## 🤖 AI Executive Summary
{ai_summary if ai_summary else 'AI analysis not available'}

## Statistics
| Metric | Value |
|--------|-------|
| Total Tools | {total} |
| Successful | {success} |
| Failed | {failed} |
| Missing | {missing} |

## Results
"""
        for r in results:
            emoji = "✅" if r.get("status") == "success" else "❌"
            md_content += f"\n### {emoji} {r.get('tool')}\n"
            md_content += f"- Status: {r.get('status')}\n"
            md_content += f"- Duration: {r.get('duration', 0):.1f}s\n"
            if r.get("ai_analysis"):
                md_content += f"- 🤖 AI: {r.get('ai_analysis')[:200]}...\n"
        
        with open(md_file, 'w') as f:
            f.write(md_content)
        
        return json_file, html_file, md_file


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AI-POWERED SCANNER
# ═══════════════════════════════════════════════════════════════════════════════
class AyedAIScanner:
    """Main AI-Powered Security Scanner"""
    
    def __init__(self, target: str = None, evidence_path: str = "/evidence"):
        self.target = target
        self.evidence_path = evidence_path
        
        # Detect system
        self.system_info = SystemDetector.detect()
        
        # Initialize AI
        self.ai = OllamaAI()
        self.ai.check_status()
        
        # Initialize installer
        self.installer = SmartInstaller(self.system_info, self.ai)
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain = urlparse(target).netloc if target and target.startswith('http') else (target or "local")
        self.output_dir = f"{Config.OUTPUT_DIR}/{domain}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize executor
        self.executor = AIToolExecutor(self.output_dir, self.ai, self.installer)
        
        # Results
        self.results = []
        
        # Set AI for logging
        Log.ai = self.ai
    
    def set_target(self, target: str):
        """Set or change target"""
        self.target = target
    
    def ai_chat(self, prompt: str) -> str:
        """Direct AI chat"""
        if self.ai and self.ai.is_available:
            return self.ai.chat(prompt)
        return "AI not available"
    
    def ai_analyze_target(self) -> str:
        """AI analyzes the target"""
        if self.ai and self.ai.is_available and self.target:
            return self.ai.analyze_target(self.target)
        return "AI not available or no target set"
    
    def run_red_team(self) -> List[Dict]:
        """Run AI-Enhanced Red Team"""
        red = AIRedTeam(self.target, self.executor, self.ai)
        results = red.run_all()
        self.results.extend(results)
        return results
    
    def run_blue_team(self) -> List[Dict]:
        """Run AI-Enhanced Blue Team"""
        blue = AIBlueTeam(self.target, self.executor, self.ai)
        results = blue.run_all()
        self.results.extend(results)
        return results
    
    def run_black_team(self) -> List[Dict]:
        """Run AI-Enhanced Black Team"""
        black = AIBlackTeam(self.target, self.executor, self.ai)
        results = black.run_all()
        self.results.extend(results)
        return results
    
    def run_forensics(self) -> List[Dict]:
        """Run AI-Enhanced Forensics"""
        forensics = AIForensics(self.evidence_path, self.executor, self.ai)
        results = forensics.run_all()
        self.results.extend(results)
        return results
    
    def run_osint(self) -> List[Dict]:
        """Run AI-Enhanced OSINT"""
        osint = AIOSINT(self.target, self.executor, self.ai)
        results = osint.run_all()
        self.results.extend(results)
        return results
    
    def run_purple_team(self) -> List[Dict]:
        """Run all teams"""
        Log.section("🟣 PURPLE TEAM - FULL AI-ENHANCED ASSESSMENT", "⚡")
        self.run_red_team()
        self.run_blue_team()
        self.run_black_team()
        self.run_osint()
        return self.results
    
    def generate_reports(self) -> Tuple[str, str, str]:
        """Generate AI-enhanced reports"""
        return AIReportGenerator.generate(self.target, self.results, self.output_dir, self.ai)
    
    def print_summary(self):
        """Print scan summary"""
        success = len([r for r in self.results if r.get("status") == "success"])
        failed = len([r for r in self.results if r.get("status") in ["failed", "error"]])
        missing = len([r for r in self.results if r.get("status") == "not_installed"])
        
        ai_status = f"{C.GRN}Online{C.R}" if self.ai.is_available else f"{C.RED}Offline{C.R}"
        
        print(f"""
{C.MAG}╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                        🤖 AI-POWERED SCAN COMPLETE 🤖                                                 ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  Target: {self.target:<95} ║
║  AI Status: {ai_status:<92} ║
║  Output: {self.output_dir:<95} ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  Total: {len(self.results):<10}  {C.GRN}✓ Success: {success:<5}{C.MAG}  {C.RED}✗ Failed: {failed:<5}{C.MAG}  {C.GRY}○ Missing: {missing:<5}{C.MAG}                     ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝{C.R}
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE CLI WITH AI
# ═══════════════════════════════════════════════════════════════════════════════
class CLI:
    scanner: AyedAIScanner = None
    current_target: str = None
    
    @staticmethod
    def clear():
        os.system('clear' if os.name != 'nt' else 'cls')
    
    @staticmethod
    def banner():
        print(f"""{C.MAG}
    ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║     █████╗ ██╗   ██╗███████╗██████╗     ██╗   ██╗ ██╗███████╗       █████╗ ██╗    ██████╗  ██████╗ ██╗    ██╗███████╗██████╗   ║
    ║    ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗    ██║   ██║███║██╔════╝      ██╔══██╗██║    ██╔══██╗██╔═══██╗██║    ██║██╔════╝██╔══██╗  ║
    ║    ███████║ ╚████╔╝ █████╗  ██║  ██║    ██║   ██║╚██║███████╗█████╗███████║██║    ██████╔╝██║   ██║██║ █╗ ██║█████╗  ██████╔╝  ║
    ║    ██╔══██║  ╚██╔╝  ██╔══╝  ██║  ██║    ╚██╗ ██╔╝ ██║╚════██║╚════╝██╔══██║██║    ██╔═══╝ ██║   ██║██║███╗██║██╔══╝  ██╔══██╗  ║
    ║    ██║  ██║   ██║   ███████╗██████╔╝     ╚████╔╝  ██║███████║      ██║  ██║██║    ██║     ╚██████╔╝╚███╔███╔╝███████╗██║  ██║  ║
    ║    ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═════╝       ╚═══╝   ╚═╝╚══════╝      ╚═╝  ╚═╝╚═╝    ╚═╝      ╚═════╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝  ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                      {C.YEL}★{C.MAG} VERSION {VERSION} - 500+ TOOLS + AI POWERED + SMART INSTALLER {C.YEL}★{C.MAG}                              ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║  {C.CYN}🤖 AI MODELS:{C.MAG} DeepSeek-R1 │ Llama3.3 │ Codestral │ Qwen2.5-Coder │ Mistral                                          ║
    ║  {C.GRN}🔧 SYSTEMS:{C.MAG}   Kali │ Parrot │ BlackArch │ Termux │ Ubuntu │ Debian │ Arch │ Fedora │ macOS                           ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝{C.R}
        """)
    
    @staticmethod
    def show_system_info():
        """Display detected system information"""
        if CLI.scanner:
            info = CLI.scanner.system_info
            ai_status = f"{C.GRN}● Online{C.R}" if CLI.scanner.ai.is_available else f"{C.RED}○ Offline{C.R}"
            models = ', '.join(CLI.scanner.ai.available_models[:3]) if CLI.scanner.ai.available_models else "None"
            
            print(f"""
    {C.CYN}╭─────────────────────────────────────────────────────────────────────╮
    │ {C.B}SYSTEM INFORMATION{C.R}{C.CYN}                                                   │
    ├─────────────────────────────────────────────────────────────────────┤
    │ {C.WHT}OS:{C.R} {info.get('name', 'Unknown'):<58}{C.CYN}│
    │ {C.WHT}Type:{C.R} {info.get('type', SystemType.UNKNOWN).value:<56}{C.CYN}│
    │ {C.WHT}Package Manager:{C.R} {info.get('package_manager', 'Unknown'):<46}{C.CYN}│
    │ {C.WHT}Root:{C.R} {'Yes' if info.get('is_root') else 'No':<58}{C.CYN}│
    ├─────────────────────────────────────────────────────────────────────┤
    │ {C.MAG}🤖 AI Status:{C.R} {ai_status:<53}{C.CYN}│
    │ {C.MAG}🤖 Models:{C.R} {models:<55}{C.CYN}│
    ╰─────────────────────────────────────────────────────────────────────╯{C.R}
            """)
    
    @staticmethod
    def main_menu():
        """Display main menu"""
        target_display = CLI.current_target[:50] if CLI.current_target else "Not Set"
        ai_icon = "🟢" if CLI.scanner and CLI.scanner.ai.is_available else "🔴"
        
        print(f"""
    {C.B}╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                          ★ MAIN MENU ★                                                            ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣{C.R}
    ║   {C.BG_RED}{C.WHT} 1 {C.R}  {C.RED}🔴 RED TEAM{C.R}        - AI-Enhanced Offensive Operations (Recon, Scanning, Web)                            ║
    ║   {C.BG_BLU}{C.WHT} 2 {C.R}  {C.BLU}🔵 BLUE TEAM{C.R}       - AI-Enhanced Defensive Operations (Hardening, Detection)                           ║
    ║   {C.BG_BLK}{C.WHT} 3 {C.R}  {C.WHT}⚫ BLACK TEAM{C.R}      - AI-Enhanced Exploitation (Password, SMB, Post-Exploit)                             ║
    ║   {C.GRN} 4 {C.R}  {C.GRN}🟢 FORENSICS{C.R}       - AI-Enhanced Investigation (Memory, Disk, Artifacts)                                 ║
    ║   {C.MAG} 5 {C.R}  {C.MAG}⚪ OSINT{C.R}           - AI-Enhanced Intelligence (Domains, Emails, Social)                                   ║
    ║   {C.MAG} 6 {C.R}  {C.MAG}🟣 PURPLE TEAM{C.R}     - Run ALL Teams with Full AI Analysis                                                  ║
    {C.B}╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣{C.R}
    ║   {C.YEL}[7]{C.R}  🎯 Set Target          {C.CYN}[8]{C.R}  {ai_icon} AI Chat            {C.GRN}[9]{C.R}  📊 System Info                               ║
    ║   {C.ORG}[I]{C.R}  🔧 Smart Installer     {C.MAG}[A]{C.R}  🤖 Setup AI Models    {C.GRY}[X]{C.R}  ❌ Exit                                      ║
    {C.B}╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝{C.R}
    
    {C.GRN}[TARGET]{C.R} {target_display}
        """)
    
    @staticmethod
    def installer_menu():
        """Smart installer menu"""
        print(f"""
    {C.ORG}╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                      🔧 SMART INSTALLER 🔧                                                        ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣{C.R}
    ║   {C.CYN}[1]{C.R}  📦 Install Core Tools       (curl, wget, git, python)                                                     ║
    ║   {C.CYN}[2]{C.R}  🔍 Install Recon Tools      (nmap, masscan, subfinder, amass)                                             ║
    ║   {C.CYN}[3]{C.R}  🌐 Install Web Tools        (nikto, sqlmap, gobuster, nuclei)                                             ║
    ║   {C.CYN}[4]{C.R}  💥 Install Exploit Tools    (hydra, john, hashcat, metasploit)                                            ║
    ║   {C.CYN}[5]{C.R}  🔬 Install Forensics Tools  (volatility, foremost, binwalk)                                               ║
    ║   {C.CYN}[6]{C.R}  🛡️  Install Defense Tools    (lynis, rkhunter, clamav, fail2ban)                                           ║
    ║   {C.MAG}[7]{C.R}  🤖 Install AI (Ollama)      (Ollama + AI Models)                                                          ║
    ║   {C.GRN}[8]{C.R}  ⚡ Install ALL Essential    (Everything above)                                                            ║
    ║   {C.YEL}[9]{C.R}  🔍 Check Missing Tools      (Scan for missing tools)                                                      ║
    ║   {C.RED}[0]{C.R}  🚀 FULL SETUP               (System Update + AI + All Tools)                                              ║
    {C.ORG}╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║   {C.GRY}[B]{C.R}  ← Back to Main Menu                                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝{C.R}
        """)
    
    @staticmethod
    def ai_menu():
        """AI configuration menu"""
        models = CLI.scanner.ai.available_models if CLI.scanner else []
        current = CLI.scanner.ai.current_model if CLI.scanner and CLI.scanner.ai else "None"
        
        print(f"""
    {C.MAG}╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                        🤖 AI CONFIGURATION 🤖                                                     ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣{C.R}
    ║   {C.CYN}[1]{C.R}  📥 Install DeepSeek-R1:7b    (Best for reasoning - 4.7GB)                                                 ║
    ║   {C.CYN}[2]{C.R}  📥 Install Llama3.2:3b       (Fast general purpose - 2GB)                                                 ║
    ║   {C.CYN}[3]{C.R}  📥 Install Codestral         (Best for security code - 12GB)                                              ║
    ║   {C.CYN}[4]{C.R}  📥 Install Qwen2.5-Coder:7b  (Alternative coder - 4.7GB)                                                  ║
    ║   {C.CYN}[5]{C.R}  📥 Install Mistral:7b        (Balanced model - 4.1GB)                                                     ║
    ║   {C.GRN}[6]{C.R}  📥 Install Top 3 Models      (DeepSeek + Llama + Codestral)                                               ║
    ║   {C.YEL}[7]{C.R}  📋 List Installed Models                                                                                  ║
    ║   {C.MAG}[8]{C.R}  🔄 Switch Model              (Change active model)                                                        ║
    ║   {C.RED}[9]{C.R}  🗑️  Remove Model              (Delete a model)                                                             ║
    {C.MAG}╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║   {C.WHT}Current Model:{C.R} {current:<80}                    ║
    ║   {C.WHT}Installed:{C.R} {', '.join(models[:5]) if models else 'None':<80}                         ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║   {C.GRY}[B]{C.R}  ← Back to Main Menu                                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝{C.R}
        """)
    
    @staticmethod
    def get_target(force_new: bool = False) -> str:
        """Get target from user"""
        if CLI.current_target and not force_new:
            print(f"\n    {C.CYN}Current: {C.GRN}{CLI.current_target}{C.R}")
            use = input(f"    {C.YEL}Use this? (Y/n): {C.R}").strip().lower()
            if use != 'n':
                return CLI.current_target
        
        target = input(f"\n    {C.YEL}Enter Target (URL/IP): {C.R}").strip()
        if target:
            if not target.startswith(('http://', 'https://')):
                target = f'https://{target}'
            CLI.current_target = target
            
            # AI target analysis
            if CLI.scanner and CLI.scanner.ai.is_available:
                print(f"\n    {C.MAG}[🤖]{C.R} Analyzing target...")
                analysis = CLI.scanner.ai.analyze_target(target)
                print(f"\n    {C.MAG}[AI ANALYSIS]{C.R}")
                print(f"    {'-' * 60}")
                for line in analysis.split('\n')[:8]:
                    print(f"    {C.GRY}{line}{C.R}")
                print(f"    {'-' * 60}")
        
        return target
    
    @staticmethod
    def ai_chat_mode():
        """Interactive AI chat"""
        if not CLI.scanner or not CLI.scanner.ai.is_available:
            print(f"\n    {C.RED}[!]{C.R} AI is not available. Install Ollama first.")
            return
        
        print(f"""
    {C.MAG}╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                        🤖 AI CHAT MODE 🤖                                                         ║
    ║                           Type 'exit' to return to main menu                                                      ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝{C.R}
        """)
        
        while True:
            try:
                prompt = input(f"\n    {C.MAG}[You]{C.R} ").strip()
                if prompt.lower() in ['exit', 'quit', 'q', 'back']:
                    break
                if not prompt:
                    continue
                
                print(f"    {C.GRY}[AI thinking...]{C.R}", end='', flush=True)
                response = CLI.scanner.ai.chat(prompt)
                print(f"\r    {C.MAG}[🤖 AI]{C.R} {response}")
                
            except KeyboardInterrupt:
                break
    
    @staticmethod
    def run_installer_menu():
        """Handle installer menu"""
        while True:
            CLI.clear()
            CLI.banner()
            CLI.installer_menu()
            
            choice = input(f"\n    {C.GRN}[?]{C.R} Select: ").strip().upper()
            
            if choice == 'B':
                return
            
            installer = CLI.scanner.installer if CLI.scanner else None
            if not installer:
                print(f"\n    {C.RED}[!]{C.R} Installer not initialized")
                time.sleep(2)
                continue
            
            if choice == '1':
                installer.install_category('core')
            elif choice == '2':
                installer.install_category('recon')
            elif choice == '3':
                installer.install_category('web')
            elif choice == '4':
                installer.install_category('exploit')
            elif choice == '5':
                installer.install_category('forensics')
            elif choice == '6':
                installer.install_category('defense')
            elif choice == '7':
                installer.install_ai_models()
            elif choice == '8':
                installer.install_all_essential()
            elif choice == '9':
                # Check missing tools
                all_tools = []
                for tools in SmartInstaller.ESSENTIAL_TOOLS.values():
                    all_tools.extend(tools)
                missing = installer.get_missing_tools(list(set(all_tools)))
                print(f"\n    {C.YEL}Missing Tools ({len(missing)}):{C.R}")
                for t in missing[:30]:
                    print(f"      - {t}")
            elif choice == '0':
                print(f"\n    {C.YEL}[!]{C.R} Starting FULL SETUP...")
                confirm = input(f"    {C.YEL}This will take a while. Continue? (y/N): {C.R}")
                if confirm.lower() == 'y':
                    installer.full_setup()
            
            input(f"\n    {C.CYN}Press Enter...{C.R}")
    
    @staticmethod
    def run_ai_menu():
        """Handle AI menu"""
        while True:
            CLI.clear()
            CLI.banner()
            CLI.ai_menu()
            
            choice = input(f"\n    {C.GRN}[?]{C.R} Select: ").strip().upper()
            
            if choice == 'B':
                return
            
            installer = CLI.scanner.installer if CLI.scanner else None
            
            if choice == '1':
                if installer:
                    subprocess.run(["ollama", "pull", "deepseek-r1:7b"])
            elif choice == '2':
                if installer:
                    subprocess.run(["ollama", "pull", "llama3.2:3b"])
            elif choice == '3':
                if installer:
                    subprocess.run(["ollama", "pull", "codestral:latest"])
            elif choice == '4':
                if installer:
                    subprocess.run(["ollama", "pull", "qwen2.5-coder:7b"])
            elif choice == '5':
                if installer:
                    subprocess.run(["ollama", "pull", "mistral:7b"])
            elif choice == '6':
                models = ["deepseek-r1:7b", "llama3.2:3b", "codestral:latest"]
                for m in models:
                    print(f"\n    {C.MAG}[AI]{C.R} Installing {m}...")
                    subprocess.run(["ollama", "pull", m])
            elif choice == '7':
                CLI.scanner.ai.check_status()
                models = CLI.scanner.ai.available_models
                print(f"\n    {C.MAG}Installed Models:{C.R}")
                for m in models:
                    print(f"      - {m}")
            elif choice == '8':
                model = input(f"    {C.YEL}Enter model name: {C.R}").strip()
                if model:
                    CLI.scanner.ai.current_model = model
                    print(f"    {C.GRN}[✓]{C.R} Model set to {model}")
            elif choice == '9':
                model = input(f"    {C.YEL}Enter model to remove: {C.R}").strip()
                if model:
                    subprocess.run(["ollama", "rm", model])
            
            input(f"\n    {C.CYN}Press Enter...{C.R}")


def interactive_menu():
    """Main interactive loop"""
    # Initialize scanner
    CLI.scanner = AyedAIScanner()
    
    while True:
        CLI.clear()
        CLI.banner()
        CLI.show_system_info()
        CLI.main_menu()
        
        choice = input(f"\n    {C.GRN}[?]{C.R} Select: ").strip().upper()
        
        if choice in ['1', '2', '3', '4', '5', '6']:
            if choice != '4' and not CLI.current_target:
                CLI.get_target()
            
            if choice != '4' and not CLI.current_target:
                continue
            
            CLI.scanner.set_target(CLI.current_target)
            
            if choice == '1':
                CLI.scanner.run_red_team()
            elif choice == '2':
                CLI.scanner.run_blue_team()
            elif choice == '3':
                CLI.scanner.run_black_team()
            elif choice == '4':
                evidence = input(f"    {C.YEL}Evidence path (default: /evidence): {C.R}").strip() or "/evidence"
                CLI.scanner.evidence_path = evidence
                CLI.scanner.run_forensics()
            elif choice == '5':
                CLI.scanner.run_osint()
            elif choice == '6':
                print(f"\n    {C.MAG}Running ALL teams with AI analysis...{C.R}")
                CLI.scanner.run_purple_team()
            
            CLI.scanner.print_summary()
            json_f, html_f, md_f = CLI.scanner.generate_reports()
            print(f"\n    {C.GRN}Reports:{C.R}")
            print(f"      📄 {json_f}")
            print(f"      🌐 {html_f}")
            print(f"      📝 {md_f}")
            
            input(f"\n    {C.CYN}Press Enter...{C.R}")
        
        elif choice == '7':
            CLI.get_target(force_new=True)
            input(f"\n    {C.CYN}Press Enter...{C.R}")
        
        elif choice == '8':
            CLI.ai_chat_mode()
        
        elif choice == '9':
            CLI.show_system_info()
            input(f"\n    {C.CYN}Press Enter...{C.R}")
        
        elif choice == 'I':
            CLI.run_installer_menu()
        
        elif choice == 'A':
            CLI.run_ai_menu()
        
        elif choice == 'X':
            print(f"\n    {C.MAG}🤖 Goodbye! Stay secure! 🔒{C.R}\n")
            sys.exit(0)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=f'Ayed v{VERSION} - AI-Powered Security Scanner')
    parser.add_argument('-t', '--target', help='Target URL/IP')
    parser.add_argument('-e', '--evidence', default='/evidence', help='Evidence path')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--red', action='store_true', help='Run Red Team')
    parser.add_argument('--blue', action='store_true', help='Run Blue Team')
    parser.add_argument('--black', action='store_true', help='Run Black Team')
    parser.add_argument('--forensics', action='store_true', help='Run Forensics')
    parser.add_argument('--osint', action='store_true', help='Run OSINT')
    parser.add_argument('--purple', action='store_true', help='Run All Teams')
    parser.add_argument('--install', action='store_true', help='Run Smart Installer')
    parser.add_argument('--setup-ai', action='store_true', help='Setup AI Models')
    parser.add_argument('--full-setup', action='store_true', help='Full System Setup')
    args = parser.parse_args()
    
    # Interactive mode if no args
    if args.interactive or len(sys.argv) == 1:
        interactive_menu()
        return
    
    # Initialize scanner
    scanner = AyedAIScanner(args.target, args.evidence)
    
    # Handle installation
    if args.full_setup:
        scanner.installer.full_setup()
        return
    
    if args.install:
        scanner.installer.install_all_essential()
        return
    
    if args.setup_ai:
        scanner.installer.install_ai_models()
        return
    
    # Run scans
    if not args.target and not args.forensics:
        parser.print_help()
        return
    
    CLI.banner = lambda: None  # Disable banner in CLI mode
    
    if args.purple:
        scanner.run_purple_team()
    else:
        if args.red:
            scanner.run_red_team()
        if args.blue:
            scanner.run_blue_team()
        if args.black:
            scanner.run_black_team()
        if args.forensics:
            scanner.run_forensics()
        if args.osint:
            scanner.run_osint()
    
    scanner.print_summary()
    json_f, html_f, md_f = scanner.generate_reports()
    print(f"\n{C.GRN}Reports:{C.R}")
    print(f"  📄 JSON: {json_f}")
    print(f"  🌐 HTML: {html_f}")
    print(f"  📝 Markdown: {md_f}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C.YEL}Interrupted.{C.R}")
    except Exception as e:
        print(f"\n{C.RED}Error: {e}{C.R}")
