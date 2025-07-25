#!/usr/bin/env python3
"""
Autonomous Backlog Management System
Implements WSJF-based prioritization and continuous backlog execution
"""

import json
import yaml
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
import subprocess
import re
import logging

@dataclass
class BacklogItem:
    """Represents a single backlog item with WSJF scoring"""
    id: str
    title: str
    type: str  # bug, feature, tech_debt, security, etc.
    description: str
    acceptance_criteria: List[str]
    effort: int  # 1-13 scale (Fibonacci-like)
    value: int   # 1-13 scale
    time_criticality: int  # 1-13 scale
    risk_reduction: int    # 1-13 scale
    status: str  # NEW, REFINED, READY, DOING, PR, DONE, BLOCKED
    risk_tier: str  # LOW, MEDIUM, HIGH
    created_at: str
    links: List[str]
    aging_days: int = 0
    
    @property
    def cost_of_delay(self) -> float:
        """Calculate Cost of Delay component of WSJF"""
        return self.value + self.time_criticality + self.risk_reduction
    
    @property 
    def wsjf_score(self) -> float:
        """Calculate WSJF score with aging multiplier"""
        if self.effort == 0:
            return 0.0
        
        base_score = self.cost_of_delay / self.effort
        aging_multiplier = min(1.0 + (self.aging_days / 30.0) * 0.5, 2.0)
        return base_score * aging_multiplier

class AutonomousBacklogManager:
    """Main autonomous backlog management system"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.backlog_file = self.repo_path / "backlog.yml"
        self.scope_file = self.repo_path / ".automation-scope.yaml"
        self.status_dir = self.repo_path / "docs" / "status"
        self.status_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
        self.backlog: List[BacklogItem] = []
        self.scope_config = self._load_scope_config()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger("autonomous_backlog")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _load_scope_config(self) -> Dict[str, Any]:
        """Load automation scope configuration"""
        if not self.scope_file.exists():
            return {}
        
        try:
            with open(self.scope_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Could not load scope config: {e}")
            return {}
    
    def discover_backlog_items(self) -> None:
        """Comprehensive backlog discovery from all sources"""
        self.logger.info("Starting comprehensive backlog discovery...")
        
        # Load existing backlog
        self._load_existing_backlog()
        
        # Discover from various sources
        self._discover_from_markdown_files()
        self._discover_from_code_comments() 
        self._discover_from_test_failures()
        self._discover_from_security_scan()
        self._discover_from_dependencies()
        
        # Update aging for existing items
        self._update_aging()
        
        # Deduplicate items by ID
        self._deduplicate_backlog()
        
        self.logger.info(f"Discovery complete. Found {len(self.backlog)} items")
    
    def _load_existing_backlog(self) -> None:
        """Load existing backlog from YAML file"""
        if not self.backlog_file.exists():
            return
            
        try:
            with open(self.backlog_file, 'r') as f:
                data = yaml.safe_load(f)
                
            # Convert YAML data to BacklogItem objects
            for status_key in ['ready', 'doing', 'pr_review', 'blocked']:
                items = data.get('backlog', {}).get(status_key, [])
                for item_data in items:
                    item = BacklogItem(**item_data)
                    self.backlog.append(item)
                    
        except Exception as e:
            self.logger.warning(f"Could not load existing backlog: {e}")
    
    def _discover_from_markdown_files(self) -> None:
        """Discover items from BACKLOG.md and other markdown files"""
        backlog_md = self.repo_path / "BACKLOG.md"
        if not backlog_md.exists():
            return
            
        try:
            content = backlog_md.read_text()
            
            # Look for incomplete items (not marked with ✅)
            incomplete_pattern = r'###\s+(\d+)\.\s+([^(]+)\(WSJF:\s*([\d.]+)\)'
            matches = re.findall(incomplete_pattern, content)
            
            for match in matches:
                item_id = f"backlog_md_{match[0]}"
                title = match[1].strip()
                wsjf_score = float(match[2])
                
                # Skip if marked as completed
                if "COMPLETED:" in title or "✅" in content:
                    continue
                
                # Reverse engineer WSJF components (rough estimation)
                effort = max(1, int(10 / max(wsjf_score, 0.1)))
                cost_of_delay = int(wsjf_score * effort)
                
                item = BacklogItem(
                    id=item_id,
                    title=title,
                    type="feature",
                    description=f"Item from BACKLOG.md: {title}",
                    acceptance_criteria=[],
                    effort=effort,
                    value=cost_of_delay // 3,
                    time_criticality=cost_of_delay // 3,
                    risk_reduction=cost_of_delay - 2 * (cost_of_delay // 3),
                    status="READY",
                    risk_tier="MEDIUM",
                    created_at=datetime.now().isoformat(),
                    links=[str(backlog_md)]
                )
                
                self.backlog.append(item)
                
        except Exception as e:
            self.logger.warning(f"Error discovering from markdown files: {e}")
    
    def _discover_from_code_comments(self) -> None:
        """Discover TODO/FIXME/XXX comments in code"""
        try:
            result = subprocess.run([
                'grep', '-r', '-n', '--include=*.py', 
                '-E', 'TODO|FIXME|XXX|HACK', 
                str(self.repo_path / 'src')
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                return  # No matches found
                
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                parts = line.split(':', 3)
                if len(parts) < 3:
                    continue
                    
                file_path = parts[0]
                line_num = parts[1]
                comment = parts[2] if len(parts) > 2 else ""
                
                item_id = f"code_comment_{hash(line) % 10000}"
                
                item = BacklogItem(
                    id=item_id,
                    title=f"Code Comment: {comment[:50]}...",
                    type="tech_debt",
                    description=f"TODO/FIXME found in {file_path}:{line_num}",
                    acceptance_criteria=[f"Resolve comment: {comment}"],
                    effort=3,  # Default small effort
                    value=2,
                    time_criticality=1,
                    risk_reduction=1,
                    status="NEW",
                    risk_tier="LOW",
                    created_at=datetime.now().isoformat(),
                    links=[f"{file_path}:{line_num}"]
                )
                
                self.backlog.append(item)
                
        except Exception as e:
            self.logger.warning(f"Error discovering code comments: {e}")
    
    def _discover_from_test_failures(self) -> None:
        """Discover items from failing tests"""
        # This would typically run pytest and parse output
        # Simplified version for now
        pass
    
    def _discover_from_security_scan(self) -> None:
        """Discover security issues via bandit scan"""
        try:
            result = subprocess.run([
                'python3', '-m', 'bandit', '-r', 'src/', '-f', 'json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                issues = data.get('results', [])
                
                for issue in issues:
                    item_id = f"security_{hash(str(issue)) % 10000}"
                    
                    severity_map = {"HIGH": 8, "MEDIUM": 5, "LOW": 3}
                    severity = severity_map.get(issue.get('issue_severity', 'LOW'), 3)
                    
                    item = BacklogItem(
                        id=item_id,
                        title=f"Security: {issue.get('test_name', 'Unknown')}",
                        type="security",
                        description=issue.get('issue_text', ''),
                        acceptance_criteria=[f"Fix security issue in {issue.get('filename', '')}"],
                        effort=3,
                        value=severity,
                        time_criticality=severity,
                        risk_reduction=severity + 2,
                        status="NEW",
                        risk_tier="HIGH" if severity >= 8 else "MEDIUM",
                        created_at=datetime.now().isoformat(),
                        links=[issue.get('filename', '')]
                    )
                    
                    self.backlog.append(item)
                    
        except Exception as e:
            self.logger.warning(f"Error running security scan: {e}")
    
    def _discover_from_dependencies(self) -> None:
        """Check for outdated project dependencies"""
        try:
            # Check project requirements files
            req_files = ['requirements.txt', 'requirements-dev.txt']
            project_deps = set()
            
            for req_file in req_files:
                req_path = self.repo_path / req_file
                if req_path.exists():
                    content = req_path.read_text()
                    for line in content.strip().split('\n'):
                        if line and not line.startswith('#'):
                            dep_name = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                            project_deps.add(dep_name.lower())
            
            if not project_deps:
                return
                
            # Get outdated packages
            result = subprocess.run([
                'python3', '-m', 'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0 and result.stdout.strip():
                outdated = json.loads(result.stdout)
                
                # Filter to only project dependencies
                outdated_project = [pkg for pkg in outdated if pkg['name'].lower() in project_deps]
                
                if outdated_project:
                    # Create refined task with clear acceptance criteria
                    critical_deps = [pkg for pkg in outdated_project if pkg['name'].lower() in ['pyjwt', 'cryptography', 'tensorflow']]
                    
                    if critical_deps:
                        # High priority for security-critical deps
                        item = BacklogItem(
                            id="security_dependency_updates",
                            title=f"Update {len(critical_deps)} Security-Critical Dependencies",
                            type="security",
                            description=f"Critical updates: {', '.join(p['name'] + ' ' + p['version'] + '->' + p['latest_version'] for p in critical_deps)}",
                            acceptance_criteria=[
                                "Update security-critical dependencies",
                                "Run full test suite to verify compatibility", 
                                "Check for breaking changes in changelog",
                                "Update requirements.txt with new versions"
                            ],
                            effort=3,
                            value=8,
                            time_criticality=6,
                            risk_reduction=8,
                            status="READY",  # Ready for execution
                            risk_tier="MEDIUM",
                            created_at=datetime.now().isoformat(),
                            links=["requirements.txt", "requirements-dev.txt"]
                        )
                    else:
                        # Lower priority for non-critical deps
                        item = BacklogItem(
                            id="dependency_maintenance",
                            title=f"Update {len(outdated_project)} Project Dependencies",
                            type="maintenance",
                            description=f"Update packages: {', '.join(p['name'] for p in outdated_project[:5])}",
                            acceptance_criteria=[
                                "Review changelog for breaking changes",
                                "Update dependencies incrementally",
                                "Run test suite after each update",
                                "Update requirements files"
                            ],
                            effort=5,
                            value=3,
                            time_criticality=2,
                            risk_reduction=4,
                            status="READY",  # Ready for execution
                            risk_tier="LOW",
                            created_at=datetime.now().isoformat(),
                            links=["requirements.txt", "requirements-dev.txt"]
                        )
                    
                    self.backlog.append(item)
                    
        except Exception as e:
            self.logger.warning(f"Error checking dependencies: {e}")
    
    def _update_aging(self) -> None:
        """Update aging days for backlog items"""
        for item in self.backlog:
            try:
                created = datetime.fromisoformat(item.created_at)
                item.aging_days = (datetime.now() - created).days
            except:
                item.aging_days = 0
    
    def _deduplicate_backlog(self) -> None:
        """Remove duplicate items based on ID"""
        seen_ids = set()
        unique_items = []
        
        for item in self.backlog:
            if item.id not in seen_ids:
                seen_ids.add(item.id)
                unique_items.append(item)
        
        self.backlog = unique_items
    
    def score_and_sort_backlog(self) -> None:
        """Score and sort backlog by WSJF"""
        self.backlog.sort(key=lambda x: x.wsjf_score, reverse=True)
        
        self.logger.info("Backlog scored and sorted by WSJF")
        for i, item in enumerate(self.backlog[:5]):
            self.logger.info(f"{i+1}. {item.title} (WSJF: {item.wsjf_score:.2f})")
    
    def save_backlog(self) -> None:
        """Save backlog to YAML file"""
        # Group items by status
        grouped = {
            'ready': [],
            'doing': [],
            'pr_review': [],
            'blocked': []
        }
        
        for item in self.backlog:
            status_key = item.status.lower().replace(' ', '_')
            if status_key in grouped:
                grouped[status_key].append(asdict(item))
        
        backlog_data = {
            'metadata': {
                'version': '2.0',
                'last_discovery': datetime.now().isoformat(),
                'total_items': len(self.backlog),
                'wsjf_methodology': True
            },
            'backlog': grouped,
            'discovery_sources': [
                'backlog_md',
                'code_comments', 
                'security_scan',
                'dependency_audit'
            ]
        }
        
        with open(self.backlog_file, 'w') as f:
            yaml.dump(backlog_data, f, default_flow_style=False, sort_keys=False)
            
        self.logger.info(f"Backlog saved to {self.backlog_file}")
    
    def generate_status_report(self) -> None:
        """Generate comprehensive status report"""
        timestamp = datetime.now().isoformat()
        
        # Count items by status
        status_counts = {}
        for item in self.backlog:
            status_counts[item.status] = status_counts.get(item.status, 0) + 1
        
        report = {
            'timestamp': timestamp,
            'mission_status': 'ACTIVE' if self.backlog else 'COMPLETE',
            'execution_summary': {
                'total_tasks_discovered': len(self.backlog),
                'wsjf_driven': True,
                'status_breakdown': status_counts
            },
            'top_priorities': [
                {
                    'id': item.id,
                    'title': item.title,
                    'wsjf_score': round(item.wsjf_score, 2),
                    'type': item.type,
                    'status': item.status
                }
                for item in self.backlog[:10]
            ],
            'system_health': {
                'backlog_size': len(self.backlog),
                'high_priority_items': len([i for i in self.backlog if i.wsjf_score > 4.0]),
                'blocked_items': len([i for i in self.backlog if i.status == 'BLOCKED']),
                'ready_items': len([i for i in self.backlog if i.status == 'READY'])
            }
        }
        
        # Save JSON report
        report_file = self.status_dir / "execution_metrics.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        self.logger.info(f"Status report saved to {report_file}")
    
    def run_discovery_cycle(self) -> None:
        """Run a complete discovery and prioritization cycle"""
        self.logger.info("Starting autonomous backlog discovery cycle...")
        
        try:
            self.discover_backlog_items()
            self.score_and_sort_backlog()
            self.save_backlog()
            self.generate_status_report()
            
            self.logger.info("Discovery cycle completed successfully")
            
        except Exception as e:
            self.logger.error(f"Discovery cycle failed: {e}")
            raise

def main():
    """Main entry point for autonomous backlog management"""
    manager = AutonomousBacklogManager()
    manager.run_discovery_cycle()
    
    if manager.backlog:
        print(f"\n🎯 Discovered {len(manager.backlog)} backlog items")
        print(f"Top priority: {manager.backlog[0].title} (WSJF: {manager.backlog[0].wsjf_score:.2f})")
    else:
        print("\n✅ No actionable backlog items found - system is complete!")

if __name__ == "__main__":
    main()