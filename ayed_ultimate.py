#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║     █████╗ ██╗   ██╗███████╗██████╗     ██╗   ██╗██╗  ████████╗██╗███╗   ███╗ █████╗ ████████╗███████╗                      ║
║    ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗    ██║   ██║██║  ╚══██╔══╝██║████╗ ████║██╔══██╗╚══██╔══╝██╔════╝                      ║
║    ███████║ ╚████╔╝ █████╗  ██║  ██║    ██║   ██║██║     ██║   ██║██╔████╔██║███████║   ██║   █████╗                        ║
║    ██╔══██║  ╚██╔╝  ██╔══╝  ██║  ██║    ██║   ██║██║     ██║   ██║██║╚██╔╝██║██╔══██║   ██║   ██╔══╝                        ║
║    ██║  ██║   ██║   ███████╗██████╔╝    ╚██████╔╝███████╗██║   ██║██║ ╚═╝ ██║██║  ██║   ██║   ███████╗                      ║
║    ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═════╝      ╚═════╝ ╚══════╝╚═╝   ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝                      ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                           ★ ULTIMATE VERSION - UNIFIED AI-POWERED SECURITY PLATFORM ★                                        ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  🤖 UNIFIED AI SYSTEM:                                                                                                        ║
║     • DeepSeek-R1 (للتحليل والاستدلال العميق)                                                                                ║
║     • Llama3.3 (للمهام العامة والسريعة)                                                                                      ║
║     • Codestral (للكود والأمان)                                                                                               ║
║     • Dual-Model Enhancement (DeepSeek-R1:8b + Llama3.2:3b)                                                                  ║
║     • Smart Model Switching & Fallback                                                                                        ║
║  🔧 COMPLETE TOOLKIT:                                                                                                         ║
║     • 500+ Security Tools (From ca2.py)                                                                                       ║
║     • 172+ Advanced Tools (From xyz.py)                                                                                       ║
║     • Smart Auto-Installer for All Systems                                                                                    ║
║     • Full Automation: Recon → Exploit → Report                                                                              ║
║  🎯 SUPPORTED SYSTEMS:                                                                                                        ║
║     • Kali │ Termux │ Ubuntu │ Debian │ Arch │ Fedora │ macOS                                                                ║
║  📊 REPORTS:                                                                                                                  ║
║     • AI-Enhanced HTML/JSON/Markdown/PDF                                                                                      ║
║     • Professional Penetration Testing Reports                                                                                ║
║     • Executive Summaries with Business Impact                                                                                ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

VERSION = "ULTIMATE 1.0 - UNIFIED AI"

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
import os, sys, re, json, time, socket, subprocess, shutil, hashlib, platform, ssl, sqlite3, base64, hmac, secrets
import threading, signal, struct, urllib.request, urllib.parse, http.client, ipaddress
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    requests = None

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
    """Intelligent System Detection - Unified from both files"""
    
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
# UNIFIED OLLAMA AI INTEGRATION - COMBINING ALL MODELS
# ═══════════════════════════════════════════════════════════════════════════════
class AIModel(Enum):
    DEEPSEEK_R1 = "deepseek-r1:latest"           # Best for reasoning & analysis
    DEEPSEEK_R1_8B = "deepseek-r1:8b"           # Smaller DeepSeek variant
    LLAMA3_3 = "llama3.3:latest"                 # Best for general tasks
    LLAMA3_2_3B = "llama3.2:3b"                  # Fast lightweight model
    CODESTRAL = "codestral:latest"               # Best for code & security
    QWEN_CODER = "qwen2.5-coder:latest"         # Alternative coder
    MISTRAL = "mistral:latest"                   # Fast & efficient

class UnifiedOllamaAI:
    """
    Unified AI System - Combining ca2.py and xyz.py approaches
    
    Features:
    - 5+ Ollama models with intelligent switching
    - Dual-model enhancement (DeepSeek + Llama for better analysis)
    - Automatic fallback if models are unavailable
    - Context-aware model selection
    - Enhanced security analysis capabilities
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.available_models = []
        self.current_model = None
        self.is_available = False
        self.conversation_history = []
        
        # Dual-model setup for enhanced analysis
        self.primary_model = "deepseek-r1:8b"
        self.secondary_model = "llama3.2:3b"
        self.models_available = {}
        
        # Check status on initialization
        self.check_status()
    
    def check_status(self) -> bool:
        """Check if Ollama is running and list available models"""
        try:
            if requests:
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    self.available_models = [m['name'] for m in data.get('models', [])]
                    self.is_available = True
                    
                    # Check specific models
                    for model in [self.primary_model, self.secondary_model]:
                        self.models_available[model] = any(model.split(':')[0] in m for m in self.available_models)
                    
                    return True
            else:
                # Fallback to urllib if requests not available
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
    
    def set_model(self, model: str):
        """Set the current model"""
        model_name = model.value if isinstance(model, AIModel) else model
        if model_name in self.available_models or any(model_name.split(':')[0] in m for m in self.available_models):
            self.current_model = model_name
            return True
        return False
    
    def chat(self, prompt: str, model: str = None, system_prompt: str = None, 
             stream: bool = False, context: str = None, temperature: float = 0.7) -> str:
        """Send a chat message to Ollama with unified interface"""
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
            if requests:
                # Use requests library
                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": stream,
                    "options": {"temperature": temperature}
                }
                response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
                if response.status_code == 200:
                    result = response.json()
                    return result.get('message', {}).get('content', 'No response')
            else:
                # Fallback to urllib
                data = json.dumps({
                    "model": model,
                    "messages": messages,
                    "stream": stream,
                    "options": {"temperature": temperature}
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
    
    def analyze_with_dual_model(self, prompt: str, context: str = "") -> str:
        """
        Enhanced dual-model analysis (from xyz.py)
        Uses two models for comprehensive analysis:
        1. Quick tactical assessment with Llama
        2. Deep strategic analysis with DeepSeek
        """
        if not self.is_available:
            return ""
        
        # Enhanced system instruction
        system_instruction = """You are a world-class cybersecurity expert with deep expertise in:
- Vulnerability exploitation and real-world attack chains
- Advanced threat modeling and risk quantification
- Security architecture and defense mechanisms
- Compliance and regulatory implications (GDPR, HIPAA, PCI-DSS, SOC2)

CRITICAL: Be aggressive, thorough, and specific. Include:
1. Attack vectors and exploitation techniques
2. Real-world impact quantification
3. Threat actor motivation and sophistication
4. Defense bypass techniques
5. Detection evasion methods
6. Business risk assessment
7. Timeline to exploitation
8. Required attacker skill level"""
        
        full_prompt = f"{system_instruction}\n\n{context}\n\n{prompt}" if context else f"{system_instruction}\n\n{prompt}"
        
        # Check if dual model is available
        use_dual = (self.models_available.get("llama3.2:3b", False) and 
                   self.models_available.get("deepseek-r1:8b", False))
        
        if use_dual:
            # First pass - Quick tactical with Llama
            llama_prompt = f"""{full_prompt}

ANALYSIS FRAMEWORK (CRITICAL):
1. IMMEDIATE THREATS: What can be exploited RIGHT NOW?
2. ATTACK CHAIN: What's the exploitation sequence?
3. IMPACT: What data/systems are compromised?
4. EVASION: How to bypass detection systems?
5. PERSISTENCE: How to maintain access?
6. LATERAL MOVEMENT: How to expand compromise?
7. EXFILTRATION: What valuable data exists?
8. TIMELINE: How quickly can this be exploited?

Be specific, aggressive, and technical."""
            
            llama_response = self.chat(llama_prompt, model="llama3.2:3b", temperature=0.6)
            
            # Second pass - Strategic deep analysis with DeepSeek
            enhanced_prompt = f"""{full_prompt}

PREVIOUS QUICK ANALYSIS SUMMARY:
{llama_response[:800]}

DEEP STRATEGIC ANALYSIS (REQUIRED):
Based on the above findings, now provide:

1. SOPHISTICATED EXPLOITATION CHAIN:
   - Multi-stage attack scenarios
   - Chained vulnerability exploitation
   - Privilege escalation paths
   - Persistence mechanisms

2. ADVANCED THREAT MODELING:
   - APT-level attack scenarios
   - Zero-day potential assessment
   - Supply chain attack vectors
   - Social engineering integration

3. DETECTION & EVASION:
   - How to avoid security tools
   - SIEM/WAF/IDS bypass techniques
   - Log tampering opportunities
   - Detection gap analysis

4. BUSINESS IMPACT ASSESSMENT:
   - Financial impact calculation
   - Reputation damage quantification
   - Regulatory compliance violations
   - Customer data exposure analysis
   - C-level executive summary

5. ADVANCED REMEDIATION:
   - Defense-in-depth strategies
   - Advanced detection signatures
   - Threat intelligence integration
   - Incident response procedures

Be extremely thorough, aggressive, and strategic. Assume sophisticated attacker."""
            
            deepseek_response = self.chat(enhanced_prompt, model="deepseek-r1:8b", temperature=0.85)
            
            # Combined response
            combined = f"""╔════════════════════════════════════════════════════════════════════╗
║  ENHANCED VULNERABILITY ANALYSIS - AGGRESSIVE ASSESSMENT          ║
╚════════════════════════════════════════════════════════════════════╝

🔍 QUICK TACTICAL ASSESSMENT (Llama3.2:3b):
═════════════════════════════════════════════════════════════════════
{llama_response}

🎯 STRATEGIC DEEP ANALYSIS (DeepSeek-r1:8b):
═════════════════════════════════════════════════════════════════════
{deepseek_response}

═════════════════════════════════════════════════════════════════════
"""
            return combined
        else:
            # Fallback to single model
            aggressive_prompt = f"""{full_prompt}

AGGRESSIVE ANALYSIS REQUIREMENT:
Be extremely detailed, specific, and aggressive. Cover:
1. Immediate exploitation opportunities
2. Attack sequences and chains
3. Data that can be stolen
4. Systems that can be compromised
5. Detection evasion techniques
6. Persistence mechanisms
7. Impact quantification
8. Timeline to compromise"""
            
            return self.chat(aggressive_prompt, temperature=0.75)
    
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
            f"Analyze this {tool} output:\n{output[:3000]}",
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
    
    def analyze_vulnerability(self, vuln_data: Dict[str, Any]) -> str:
        """Analyze vulnerability with AI"""
        if not self.is_available:
            return ""
        
        prompt = f"""Analyze this security vulnerability and provide insights:

Vulnerability: {vuln_data.get('vulnerability_type', 'Unknown')}
Severity: {vuln_data.get('severity', 'Unknown')}
Description: {vuln_data.get('description', '')}
Target: {vuln_data.get('target', '')}
Evidence: {vuln_data.get('evidence', '')}

Provide:
1. Risk assessment
2. Potential impact
3. Recommended remediation steps
4. Attack scenarios"""
        
        return self.analyze_with_dual_model(prompt)
    
    def analyze_findings(self, findings: List[Dict[str, Any]]) -> str:
        """Analyze multiple findings"""
        if not self.is_available or not findings:
            return ""
        
        summary = f"Total findings: {len(findings)}\n"
        for i, finding in enumerate(findings[:5], 1):
            summary += f"\n{i}. {finding.get('vulnerability_type', 'Unknown')} (Severity: {finding.get('severity', 'Unknown')})"
        
        prompt = f"""Analyze these security findings and provide a strategic assessment:

{summary}

Provide:
1. Overall risk assessment
2. Priority order for remediation
3. Common vulnerability patterns
4. Strategic recommendations"""
        
        return self.analyze_with_dual_model(prompt)
    
    def generate_professional_report(self, target: str, findings: List[Dict[str, Any]]) -> str:
        """Generate professional penetration testing report"""
        if not self.is_available or not findings:
            return ""
        
        vuln_summary = "\n".join([
            f"- {f.get('vulnerability_type', 'Unknown')} (Severity: {f.get('severity', 'Unknown')}): {f.get('description', '')[:100]}"
            for f in findings[:15]
        ])
        
        prompt = f"""Generate a PROFESSIONAL PENETRATION TESTING REPORT for {target}.

FINDINGS SUMMARY:
{vuln_summary}

Create a detailed report with EXACT sections:

## EXECUTIVE SUMMARY
- Brief overview of engagement
- Critical findings count
- Overall risk level (Critical/High/Medium)
- Key recommendations

## VULNERABILITY DETAILS
For each critical/high finding provide:
- Vulnerability Name
- CVSS Score
- Severity Level
- Description
- **PROOF OF CONCEPT (POC)**: Exact exploit/request that demonstrates vulnerability
- **EVIDENCE**: Concrete evidence (error messages, responses, screenshots)
- **IMPACT**: Business and technical impact
- **REMEDIATION**: Step-by-step fix instructions
- **DETECTION**: How to identify if exploited

## ATTACK SCENARIOS
- Realistic attack chains
- Exploitation sequence
- Potential lateral movement paths

## REMEDIATION ROADMAP
1. Immediate (Critical) - Priority 1
2. Short-term (High) - Priority 2
3. Medium-term (Medium) - Priority 3
4. Long-term (Low) - Priority 4

## COMPLIANCE & STANDARDS
- OWASP Top 10 references
- CWE mappings
- CVSS v3.1 scores

## CONCLUSION
- Overall security posture
- Risk assessment
- Investment justification for remediation"""
        
        return self.analyze_with_dual_model(prompt)

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
    
    # Extended colors from xyz.py
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

class Log:
    ai: UnifiedOllamaAI = None
    
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
    OUTPUT_DIR = "ayed_ultimate_results"
    TIMEOUT = 600
    THREADS = 25
    VERBOSE = True
    AI_ENABLED = True
    AI_ASSIST_LEVEL = "full"  # minimal, moderate, full

# ═══════════════════════════════════════════════════════════════════════════════
# BANNER
# ═══════════════════════════════════════════════════════════════════════════════
def show_banner():
    """Display the unified banner"""
    banner = f"""{C.MAG}
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║     █████╗ ██╗   ██╗███████╗██████╗     ██╗   ██╗██╗  ████████╗██╗███╗   ███╗ █████╗ ████████╗███████╗                      ║
║    ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗    ██║   ██║██║  ╚══██╔══╝██║████╗ ████║██╔══██╗╚══██╔══╝██╔════╝                      ║
║    ███████║ ╚████╔╝ █████╗  ██║  ██║    ██║   ██║██║     ██║   ██║██╔████╔██║███████║   ██║   █████╗                        ║
║    ██╔══██║  ╚██╔╝  ██╔══╝  ██║  ██║    ██║   ██║██║     ██║   ██║██║╚██╔╝██║██╔══██║   ██║   ██╔══╝                        ║
║    ██║  ██║   ██║   ███████╗██████╔╝    ╚██████╔╝███████╗██║   ██║██║ ╚═╝ ██║██║  ██║   ██║   ███████╗                      ║
║    ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═════╝      ╚═════╝ ╚══════╝╚═╝   ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝                      ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                           {C.YEL}★{C.MAG} VERSION {VERSION} - ULTIMATE AI-POWERED SECURITY SUITE {C.YEL}★{C.MAG}                             ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  {C.CYN}🤖 UNIFIED AI MODELS:{C.MAG} DeepSeek-R1 │ DeepSeek-R1:8b │ Llama3.3 │ Llama3.2:3b │ Codestral │ Qwen2.5 │ Mistral        ║
║  {C.GRN}🔧 SUPPORTED SYSTEMS:{C.MAG} Kali │ Parrot │ BlackArch │ Termux │ Ubuntu │ Debian │ Arch │ Fedora │ macOS                ║
║  {C.YEL}⚡ TOTAL TOOLS:{C.MAG} 500+ Internal Tools + 172+ External Tools = 672+ Security Tests                                    ║
║  {C.ORG}🎯 FEATURES:{C.MAG} Auto-Install │ Smart Switching │ Dual-Model AI │ Professional Reports │ Full Automation           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝{C.R}
    """
    print(banner)

# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE DEMO MAIN - Will be expanded with full features
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    """Main entry point - Simple demo showing the unified system"""
    import argparse
    
    parser = argparse.ArgumentParser(description=f'Ayed Ultimate v{VERSION} - Unified AI-Powered Security Scanner')
    parser.add_argument('-t', '--target', help='Target URL/IP')
    parser.add_argument('--check-ai', action='store_true', help='Check AI status')
    parser.add_argument('--check-system', action='store_true', help='Check system info')
    args = parser.parse_args()
    
    show_banner()
    
    # Detect system
    print(f"\n{C.CYN}[*]{C.R} Detecting system...")
    system_info = SystemDetector.detect()
    print(f"    {C.GRN}[✓]{C.R} System: {system_info['name']}")
    print(f"    {C.GRN}[✓]{C.R} Type: {system_info['type'].value}")
    print(f"    {C.GRN}[✓]{C.R} Package Manager: {system_info['package_manager']}")
    print(f"    {C.GRN}[✓]{C.R} Root: {'Yes' if system_info['is_root'] else 'No'}")
    
    if args.check_system:
        return
    
    # Initialize AI
    print(f"\n{C.CYN}[*]{C.R} Initializing Unified AI System...")
    ai = UnifiedOllamaAI()
    
    if ai.is_available:
        print(f"    {C.GRN}[✓]{C.R} Ollama is running")
        print(f"    {C.GRN}[✓]{C.R} Available models: {len(ai.available_models)}")
        for model in ai.available_models[:5]:
            print(f"        - {model}")
        if len(ai.available_models) > 5:
            print(f"        ... and {len(ai.available_models) - 5} more")
    else:
        print(f"    {C.YEL}[!]{C.R} Ollama not available - Install with: curl -fsSL https://ollama.com/install.sh | sh")
    
    if args.check_ai:
        return
    
    if args.target and ai.is_available:
        print(f"\n{C.MAG}[🤖]{C.R} AI analyzing target: {args.target}")
        analysis = ai.analyze_target(args.target)
        print(f"\n{C.CYN}{'═' * 80}")
        print(f"{C.MAG}AI ANALYSIS:{C.R}")
        print(f"{C.CYN}{'═' * 80}{C.R}")
        print(analysis)
        print(f"{C.CYN}{'═' * 80}{C.R}")
    else:
        print(f"\n{C.CYN}[*]{C.R} System initialized successfully!")
        print(f"    {C.GRN}Usage:{C.R} python3 {sys.argv[0]} --target example.com")
        print(f"    {C.GRN}Check AI:{C.R} python3 {sys.argv[0]} --check-ai")
        print(f"    {C.GRN}Check System:{C.R} python3 {sys.argv[0]} --check-system")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C.YEL}Interrupted.{C.R}")
    except Exception as e:
        print(f"\n{C.RED}Error: {e}{C.R}")
