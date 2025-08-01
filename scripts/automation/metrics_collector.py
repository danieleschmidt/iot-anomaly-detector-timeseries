#!/usr/bin/env python3
"""
SDLC Metrics Collector

This script collects comprehensive SDLC metrics and updates the project metrics file.
It integrates with various tools and services to gather real-time metrics.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests
import yaml


class SDLCMetricsCollector:
    """Comprehensive SDLC metrics collector."""
    
    def __init__(self, repo_root: Path = None, config_path: Path = None):
        self.repo_root = repo_root or Path.cwd()
        self.config_path = config_path or (self.repo_root / '.github' / 'project-metrics.json')
        self.metrics = self.load_existing_metrics()
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.repo_name = self.get_repository_name()
    
    def load_existing_metrics(self) -> Dict[str, Any]:
        """Load existing metrics from project metrics file."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)
        else:
            return {
                "project": {},
                "sdlc_metrics": {},
                "quality_gates": {},
                "automation_status": {},
                "development_metrics": {},
                "technical_debt": {},
                "dependencies": {},
                "monitoring": {}
            }
    
    def get_repository_name(self) -> str:
        """Get repository name from git remote."""
        try:
            result = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
                capture_output=True, text=True, cwd=self.repo_root
            )
            if result.returncode == 0:
                url = result.stdout.strip()
                # Extract owner/repo from various URL formats
                if 'github.com' in url:
                    if url.endswith('.git'):
                        url = url[:-4]
                    return url.split('/')[-2] + '/' + url.split('/')[-1]
        except Exception:
            pass
        
        return "unknown/repository"
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all SDLC metrics."""
        print("📊 Collecting SDLC metrics...")
        
        # Update timestamp
        self.metrics['sdlc_metrics']['last_updated'] = datetime.now().isoformat()
        
        # Collect various metrics
        self.collect_code_quality_metrics()
        self.collect_testing_metrics()
        self.collect_security_metrics()
        self.collect_dependency_metrics()
        self.collect_cicd_metrics()
        self.collect_repository_metrics()
        self.collect_performance_metrics()
        self.calculate_composite_scores()
        
        print("✅ Metrics collection completed!")
        return self.metrics
    
    def collect_code_quality_metrics(self):
        """Collect code quality metrics."""
        print("🔍 Collecting code quality metrics...")
        
        quality_metrics = {}
        
        # Run ruff for code quality
        try:
            result = subprocess.run(
                ['ruff', 'check', '.', '--format=json'],
                capture_output=True, text=True, cwd=self.repo_root
            )
            
            if result.stdout:
                ruff_data = json.loads(result.stdout)
                quality_metrics['ruff_issues'] = len(ruff_data)
                
                # Categorize issues
                issue_types = {}
                for issue in ruff_data:
                    rule_code = issue.get('code', 'unknown')
                    issue_types[rule_code] = issue_types.get(rule_code, 0) + 1
                
                quality_metrics['ruff_issue_types'] = issue_types
            else:
                quality_metrics['ruff_issues'] = 0
                
        except (FileNotFoundError, json.JSONDecodeError, subprocess.CalledProcessError):
            quality_metrics['ruff_issues'] = 'N/A'
        
        # Run mypy for type checking
        try:
            result = subprocess.run(
                ['mypy', 'src/', '--json-report', '/tmp/mypy-report'],
                capture_output=True, text=True, cwd=self.repo_root
            )
            
            mypy_report_path = Path('/tmp/mypy-report/index.txt')
            if mypy_report_path.exists():
                # Parse mypy report (simplified)
                quality_metrics['mypy_errors'] = result.stdout.count('error:')
                quality_metrics['mypy_warnings'] = result.stdout.count('warning:')
            
        except (FileNotFoundError, subprocess.CalledProcessError):
            quality_metrics['mypy_errors'] = 'N/A'
        
        # Count lines of code
        try:
            src_files = list((self.repo_root / 'src').rglob('*.py'))
            total_lines = 0
            total_files = len(src_files)
            
            for file_path in src_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        # Count non-empty, non-comment lines
                        code_lines = [line for line in lines 
                                    if line.strip() and not line.strip().startswith('#')]
                        total_lines += len(code_lines)
                except UnicodeDecodeError:
                    continue
            
            quality_metrics['lines_of_code'] = total_lines
            quality_metrics['python_files'] = total_files
            
            if total_lines > 0:
                quality_metrics['issues_per_kloc'] = (quality_metrics.get('ruff_issues', 0) / total_lines) * 1000
            
        except Exception:
            quality_metrics['lines_of_code'] = 'N/A'
        
        self.metrics['quality_gates']['code_quality_metrics'] = quality_metrics
    
    def collect_testing_metrics(self):
        """Collect testing and coverage metrics."""
        print("🧪 Collecting testing metrics...")
        
        testing_metrics = {}
        
        # Run pytest with coverage
        try:
            result = subprocess.run(
                ['python', '-m', 'pytest', '--cov=src', '--cov-report=json', '--cov-report=term'],
                capture_output=True, text=True, cwd=self.repo_root
            )
            
            # Parse coverage report
            coverage_file = self.repo_root / 'coverage.json'
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                
                testing_metrics['line_coverage'] = coverage_data['totals']['percent_covered']
                testing_metrics['lines_covered'] = coverage_data['totals']['covered_lines']
                testing_metrics['lines_missing'] = coverage_data['totals']['missing_lines']
                testing_metrics['total_lines'] = coverage_data['totals']['num_statements']
                
                # File-level coverage
                file_coverage = {}
                for filename, file_data in coverage_data['files'].items():
                    file_coverage[filename] = file_data['summary']['percent_covered']
                
                testing_metrics['file_coverage'] = file_coverage
                
                # Find files with low coverage
                low_coverage_files = {
                    filename: coverage 
                    for filename, coverage in file_coverage.items() 
                    if coverage < 80
                }
                testing_metrics['low_coverage_files'] = low_coverage_files
            
            # Parse test results
            if 'passed' in result.stdout:
                # Extract test counts (simplified parsing)
                output_lines = result.stdout.split('\\n')
                for line in output_lines:
                    if 'passed' in line and 'failed' in line:
                        # Parse line like "10 passed, 2 failed"
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'passed' and i > 0:
                                testing_metrics['tests_passed'] = int(parts[i-1])
                            elif part == 'failed' and i > 0:
                                testing_metrics['tests_failed'] = int(parts[i-1])
                        break
                
        except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
            testing_metrics['line_coverage'] = 'N/A'
        
        # Count test files
        tests_dir = self.repo_root / 'tests'
        if tests_dir.exists():
            test_files = list(tests_dir.rglob('test_*.py'))
            testing_metrics['test_files_count'] = len(test_files)
            
            # Categorize tests
            test_categories = {
                'unit': 0,
                'integration': 0,
                'e2e': 0,
                'performance': 0,
                'security': 0
            }
            
            for test_file in test_files:
                filename = test_file.name.lower()
                if 'integration' in filename:
                    test_categories['integration'] += 1
                elif 'e2e' in filename or 'end_to_end' in filename:
                    test_categories['e2e'] += 1
                elif 'performance' in filename or 'benchmark' in filename:
                    test_categories['performance'] += 1
                elif 'security' in filename:
                    test_categories['security'] += 1
                else:
                    test_categories['unit'] += 1
            
            testing_metrics['test_categories'] = test_categories
        
        self.metrics['quality_gates']['testing_metrics'] = testing_metrics
    
    def collect_security_metrics(self):
        """Collect security metrics."""
        print("🔒 Collecting security metrics...")
        
        security_metrics = {}
        
        # Run bandit for security issues
        try:
            result = subprocess.run(
                ['bandit', '-r', 'src/', '-f', 'json'],
                capture_output=True, text=True, cwd=self.repo_root
            )
            
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                
                security_metrics['bandit_issues'] = len(bandit_data.get('results', []))
                
                # Categorize by severity
                severity_counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
                for issue in bandit_data.get('results', []):
                    severity = issue.get('issue_severity', 'UNKNOWN')
                    if severity in severity_counts:
                        severity_counts[severity] += 1
                
                security_metrics['security_issues_by_severity'] = severity_counts
                
                # Most common issue types
                issue_types = {}
                for issue in bandit_data.get('results', []):
                    test_id = issue.get('test_id', 'unknown')
                    issue_types[test_id] = issue_types.get(test_id, 0) + 1
                
                security_metrics['common_security_issues'] = dict(
                    sorted(issue_types.items(), key=lambda x: x[1], reverse=True)[:5]
                )
            
        except (FileNotFoundError, json.JSONDecodeError, subprocess.CalledProcessError):
            security_metrics['bandit_issues'] = 'N/A'
        
        # Run safety for dependency vulnerabilities
        try:
            result = subprocess.run(
                ['safety', 'check', '--json'],
                capture_output=True, text=True, cwd=self.repo_root
            )
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                security_metrics['vulnerable_dependencies'] = len(safety_data)
                
                # Extract vulnerability details
                vulnerabilities = []
                for vuln in safety_data:
                    vulnerabilities.append({
                        'package': vuln.get('package'),
                        'installed_version': vuln.get('installed_version'),
                        'vulnerability_id': vuln.get('vulnerability_id'),
                        'severity': vuln.get('severity', 'unknown')
                    })
                
                security_metrics['vulnerability_details'] = vulnerabilities
            
        except (FileNotFoundError, json.JSONDecodeError, subprocess.CalledProcessError):
            security_metrics['vulnerable_dependencies'] = 'N/A'
        
        # Check for security configuration files
        security_files = [
            'SECURITY.md',
            '.github/dependabot.yml',
            '.pre-commit-config.yaml'
        ]
        
        existing_security_files = []
        for sec_file in security_files:
            if (self.repo_root / sec_file).exists():
                existing_security_files.append(sec_file)
        
        security_metrics['security_files_present'] = existing_security_files
        security_metrics['security_config_completeness'] = len(existing_security_files) / len(security_files)
        
        self.metrics['quality_gates']['security_metrics'] = security_metrics
    
    def collect_dependency_metrics(self):
        """Collect dependency management metrics."""
        print("📦 Collecting dependency metrics...")
        
        dependency_metrics = {}
        
        # Parse requirements.txt
        req_path = self.repo_root / 'requirements.txt'
        if req_path.exists():
            requirements = req_path.read_text()
            lines = [line.strip() for line in requirements.split('\\n') 
                    if line.strip() and not line.startswith('#')]
            
            dependency_metrics['total_dependencies'] = len(lines)
            
            # Check for version pinning
            pinned_count = 0
            for line in lines:
                if any(op in line for op in ['==', '>=', '<=', '~=', '!=']):
                    pinned_count += 1
            
            dependency_metrics['pinned_dependencies'] = pinned_count
            dependency_metrics['pinning_ratio'] = pinned_count / len(lines) if lines else 0
            
            # Extract package names
            packages = []
            for line in lines:
                # Extract package name (before any version specifiers)
                package_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].split('!=')[0].strip()
                packages.append(package_name)
            
            dependency_metrics['package_list'] = packages
        
        # Check for development dependencies
        req_dev_path = self.repo_root / 'requirements-dev.txt'
        if req_dev_path.exists():
            dev_requirements = req_dev_path.read_text()
            dev_lines = [line.strip() for line in dev_requirements.split('\\n') 
                        if line.strip() and not line.startswith('#')]
            dependency_metrics['dev_dependencies'] = len(dev_lines)
        
        # Check pyproject.toml
        pyproject_path = self.repo_root / 'pyproject.toml'
        if pyproject_path.exists():
            try:
                import toml
                pyproject_data = toml.load(pyproject_path)
                
                if 'project' in pyproject_data and 'dependencies' in pyproject_data['project']:
                    pyproject_deps = len(pyproject_data['project']['dependencies'])
                    dependency_metrics['pyproject_dependencies'] = pyproject_deps
                
                if 'project' in pyproject_data and 'optional-dependencies' in pyproject_data['project']:
                    optional_deps = pyproject_data['project']['optional-dependencies']
                    dependency_metrics['optional_dependencies'] = {
                        group: len(deps) for group, deps in optional_deps.items()
                    }
                
            except ImportError:
                dependency_metrics['pyproject_dependencies'] = 'N/A (toml not available)'
        
        # Check for outdated dependencies
        try:
            result = subprocess.run(
                ['pip', 'list', '--outdated', '--format=json'],
                capture_output=True, text=True, cwd=self.repo_root
            )
            
            if result.stdout:
                outdated_data = json.loads(result.stdout)
                dependency_metrics['outdated_dependencies'] = len(outdated_data)
                
                # Extract outdated package details
                outdated_packages = []
                for package in outdated_data:
                    outdated_packages.append({
                        'name': package['name'],
                        'current_version': package['version'],
                        'latest_version': package['latest_version']
                    })
                
                dependency_metrics['outdated_package_details'] = outdated_packages
            
        except (FileNotFoundError, json.JSONDecodeError, subprocess.CalledProcessError):
            dependency_metrics['outdated_dependencies'] = 'N/A'
        
        self.metrics['dependencies'] = dependency_metrics
    
    def collect_cicd_metrics(self):
        """Collect CI/CD pipeline metrics."""
        print("⚙️ Collecting CI/CD metrics...")
        
        cicd_metrics = {}
        
        # Check workflow files
        workflows_dir = self.repo_root / '.github' / 'workflows'
        if workflows_dir.exists():
            workflow_files = list(workflows_dir.glob('*.yml'))
            cicd_metrics['workflow_count'] = len(workflow_files)
            
            workflow_types = []
            for workflow_file in workflow_files:
                workflow_types.append(workflow_file.stem)
            
            cicd_metrics['workflow_types'] = workflow_types
            
            # Analyze workflow complexity
            total_jobs = 0
            total_steps = 0
            
            for workflow_file in workflow_files:
                try:
                    with open(workflow_file) as f:
                        workflow_data = yaml.safe_load(f)
                    
                    if 'jobs' in workflow_data:
                        total_jobs += len(workflow_data['jobs'])
                        
                        for job_name, job_data in workflow_data['jobs'].items():
                            if 'steps' in job_data:
                                total_steps += len(job_data['steps'])
                
                except yaml.YAMLError:
                    continue
            
            cicd_metrics['total_jobs'] = total_jobs
            cicd_metrics['total_steps'] = total_steps
            cicd_metrics['avg_steps_per_workflow'] = total_steps / len(workflow_files) if workflow_files else 0
        else:
            cicd_metrics['workflow_count'] = 0
        
        # GitHub API metrics (if token available)
        if self.github_token and self.repo_name != "unknown/repository":
            try:
                headers = {
                    'Authorization': f'token {self.github_token}',
                    'Accept': 'application/vnd.github.v3+json'
                }
                
                # Get recent workflow runs
                runs_url = f'https://api.github.com/repos/{self.repo_name}/actions/runs'
                response = requests.get(runs_url, headers=headers, params={'per_page': 20})
                
                if response.status_code == 200:
                    runs_data = response.json()
                    
                    success_count = 0
                    failure_count = 0
                    total_duration = 0
                    
                    for run in runs_data.get('workflow_runs', []):
                        if run['conclusion'] == 'success':
                            success_count += 1
                        elif run['conclusion'] == 'failure':
                            failure_count += 1
                        
                        # Calculate duration if timestamps available
                        if run['created_at'] and run['updated_at']:
                            created = datetime.fromisoformat(run['created_at'].replace('Z', '+00:00'))
                            updated = datetime.fromisoformat(run['updated_at'].replace('Z', '+00:00'))
                            duration = (updated - created).total_seconds()
                            total_duration += duration
                    
                    total_runs = len(runs_data.get('workflow_runs', []))
                    if total_runs > 0:
                        cicd_metrics['success_rate'] = success_count / total_runs
                        cicd_metrics['failure_rate'] = failure_count / total_runs
                        cicd_metrics['avg_duration_seconds'] = total_duration / total_runs
                    
                    cicd_metrics['recent_runs_analyzed'] = total_runs
                
            except requests.RequestException:
                cicd_metrics['github_api_metrics'] = 'N/A (API unavailable)'
        
        self.metrics['automation_status']['cicd_metrics'] = cicd_metrics
    
    def collect_repository_metrics(self):
        """Collect general repository metrics."""
        print("📁 Collecting repository metrics...")
        
        repo_metrics = {}
        
        # Git statistics
        try:
            # Count commits
            result = subprocess.run(
                ['git', 'rev-list', '--count', 'HEAD'],
                capture_output=True, text=True, cwd=self.repo_root
            )
            if result.returncode == 0:
                repo_metrics['total_commits'] = int(result.stdout.strip())
            
            # Get contributors
            result = subprocess.run(
                ['git', 'shortlog', '-sn'],
                capture_output=True, text=True, cwd=self.repo_root
            )
            if result.returncode == 0:
                contributors = result.stdout.strip().split('\\n')
                repo_metrics['contributors_count'] = len(contributors)
            
            # Get recent activity
            one_month_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            result = subprocess.run(
                ['git', 'rev-list', '--count', f'--since={one_month_ago}', 'HEAD'],
                capture_output=True, text=True, cwd=self.repo_root
            )
            if result.returncode == 0:
                repo_metrics['commits_last_month'] = int(result.stdout.strip())
            
        except (subprocess.CalledProcessError, ValueError):
            repo_metrics['git_stats'] = 'N/A'
        
        # File and directory counts
        try:
            # Count Python files
            python_files = list(self.repo_root.rglob('*.py'))
            repo_metrics['python_files'] = len(python_files)
            
            # Count documentation files
            doc_files = list(self.repo_root.rglob('*.md'))
            repo_metrics['documentation_files'] = len(doc_files)
            
            # Count configuration files
            config_extensions = ['.yml', '.yaml', '.json', '.toml', '.ini', '.cfg']
            config_files = []
            for ext in config_extensions:
                config_files.extend(list(self.repo_root.rglob(f'*{ext}')))
            
            repo_metrics['configuration_files'] = len(config_files)
            
        except Exception:
            repo_metrics['file_counts'] = 'N/A'
        
        # Repository size
        try:
            result = subprocess.run(
                ['du', '-sh', '.'],
                capture_output=True, text=True, cwd=self.repo_root
            )
            if result.returncode == 0:
                size_str = result.stdout.split()[0]
                repo_metrics['repository_size'] = size_str
        except subprocess.CalledProcessError:
            repo_metrics['repository_size'] = 'N/A'
        
        self.metrics['development_metrics']['repository_stats'] = repo_metrics
    
    def collect_performance_metrics(self):
        """Collect performance-related metrics."""
        print("⚡ Collecting performance metrics...")
        
        perf_metrics = {}
        
        # Check if there are performance test files
        perf_test_files = []
        for pattern in ['*performance*', '*benchmark*', '*load*']:
            perf_test_files.extend(list(self.repo_root.rglob(f'{pattern}.py')))
        
        perf_metrics['performance_test_files'] = len(perf_test_files)
        
        # Look for performance monitoring configuration
        monitoring_files = [
            'config/monitoring/prometheus.yml',
            'config/grafana-dashboard.json',
            'docker-compose.yml'
        ]
        
        monitoring_present = []
        for mon_file in monitoring_files:
            if (self.repo_root / mon_file).exists():
                monitoring_present.append(mon_file)
        
        perf_metrics['monitoring_files_present'] = monitoring_present
        perf_metrics['monitoring_completeness'] = len(monitoring_present) / len(monitoring_files)
        
        # Check for health check endpoints in code
        health_check_found = False
        if (self.repo_root / 'src').exists():
            for py_file in (self.repo_root / 'src').rglob('*.py'):
                try:
                    content = py_file.read_text().lower()
                    if 'health' in content and ('endpoint' in content or 'route' in content):
                        health_check_found = True
                        break
                except UnicodeDecodeError:
                    continue
        
        perf_metrics['health_check_implemented'] = health_check_found
        
        # Check Docker optimization
        dockerfile_path = self.repo_root / 'Dockerfile'
        if dockerfile_path.exists():
            dockerfile_content = dockerfile_path.read_text().lower()
            
            optimization_features = {
                'multi_stage_build': 'from' in dockerfile_content and 'as' in dockerfile_content,
                'layer_caching': '.dockerignore' in os.listdir(self.repo_root),
                'non_root_user': 'user' in dockerfile_content,
                'health_check': 'healthcheck' in dockerfile_content
            }
            
            perf_metrics['docker_optimizations'] = optimization_features
            perf_metrics['docker_optimization_score'] = sum(optimization_features.values()) / len(optimization_features)
        
        self.metrics['monitoring']['performance_metrics'] = perf_metrics
    
    def calculate_composite_scores(self):
        """Calculate composite SDLC scores."""
        print("📈 Calculating composite scores...")
        
        # Calculate overall SDLC completeness
        components = {
            'code_quality': self.get_code_quality_score(),
            'testing': self.get_testing_score(),
            'security': self.get_security_score(),
            'automation': self.get_automation_score(),
            'documentation': self.get_documentation_score(),
            'monitoring': self.get_monitoring_score()
        }
        
        # Filter out N/A values
        valid_scores = {k: v for k, v in components.items() if isinstance(v, (int, float))}
        
        if valid_scores:
            overall_score = sum(valid_scores.values()) / len(valid_scores)
        else:
            overall_score = 0
        
        self.metrics['sdlc_metrics'].update({
            'completeness_score': round(overall_score, 2),
            'component_scores': components,
            'last_calculated': datetime.now().isoformat()
        })
    
    def get_code_quality_score(self) -> float:
        """Calculate code quality score."""
        quality_metrics = self.metrics.get('quality_gates', {}).get('code_quality_metrics', {})
        
        score = 100
        
        # Deduct points for issues
        ruff_issues = quality_metrics.get('ruff_issues', 0)
        if isinstance(ruff_issues, int):
            # Deduct 1 point per issue, max 30 points
            score -= min(ruff_issues, 30)
        
        mypy_errors = quality_metrics.get('mypy_errors', 0)
        if isinstance(mypy_errors, int):
            # Deduct 2 points per error, max 20 points
            score -= min(mypy_errors * 2, 20)
        
        return max(score, 0)
    
    def get_testing_score(self) -> float:
        """Calculate testing score."""
        testing_metrics = self.metrics.get('quality_gates', {}).get('testing_metrics', {})
        
        coverage = testing_metrics.get('line_coverage', 0)
        if isinstance(coverage, (int, float)):
            return min(coverage, 100)
        
        return 0
    
    def get_security_score(self) -> float:
        """Calculate security score."""
        security_metrics = self.metrics.get('quality_gates', {}).get('security_metrics', {})
        
        score = 100
        
        # Deduct for security issues
        bandit_issues = security_metrics.get('bandit_issues', 0)
        if isinstance(bandit_issues, int):
            score -= min(bandit_issues * 5, 50)  # 5 points per issue, max 50
        
        vulnerable_deps = security_metrics.get('vulnerable_dependencies', 0)
        if isinstance(vulnerable_deps, int):
            score -= min(vulnerable_deps * 10, 40)  # 10 points per vuln, max 40
        
        # Add points for security configuration
        config_completeness = security_metrics.get('security_config_completeness', 0)
        if isinstance(config_completeness, (int, float)):
            score = max(score, config_completeness * 100)
        
        return max(score, 0)
    
    def get_automation_score(self) -> float:
        """Calculate automation score."""
        cicd_metrics = self.metrics.get('automation_status', {}).get('cicd_metrics', {})
        
        workflow_count = cicd_metrics.get('workflow_count', 0)
        
        if workflow_count >= 3:
            score = 100
        elif workflow_count >= 2:
            score = 80
        elif workflow_count >= 1:
            score = 60
        else:
            score = 0
        
        # Adjust based on success rate if available
        success_rate = cicd_metrics.get('success_rate')
        if isinstance(success_rate, (int, float)):
            score *= success_rate
        
        return score
    
    def get_documentation_score(self) -> float:
        """Calculate documentation score."""
        repo_stats = self.metrics.get('development_metrics', {}).get('repository_stats', {})
        
        doc_files = repo_stats.get('documentation_files', 0)
        
        if doc_files >= 10:
            return 100
        elif doc_files >= 5:
            return 80
        elif doc_files >= 3:
            return 60
        elif doc_files >= 1:
            return 40
        else:
            return 0
    
    def get_monitoring_score(self) -> float:
        """Calculate monitoring score."""
        perf_metrics = self.metrics.get('monitoring', {}).get('performance_metrics', {})
        
        completeness = perf_metrics.get('monitoring_completeness', 0)
        health_check = perf_metrics.get('health_check_implemented', False)
        
        score = completeness * 70  # 70% for monitoring files
        if health_check:
            score += 30  # 30% for health check
        
        return min(score * 100, 100)  # Convert to percentage
    
    def save_metrics(self, output_path: Path = None):
        """Save collected metrics to file."""
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"💾 Metrics saved to: {output_path}")
    
    def print_summary(self):
        """Print metrics summary."""
        print(f"\\n📊 SDLC Metrics Summary")
        print("=" * 50)
        
        completeness_score = self.metrics.get('sdlc_metrics', {}).get('completeness_score', 0)
        print(f"Overall SDLC Score: {completeness_score:.1f}%")
        
        component_scores = self.metrics.get('sdlc_metrics', {}).get('component_scores', {})
        
        print(f"\\n🎯 Component Scores:")
        for component, score in component_scores.items():
            if isinstance(score, (int, float)):
                emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
                print(f"  {emoji} {component.replace('_', ' ').title()}: {score:.1f}%")
            else:
                print(f"  ⚪ {component.replace('_', ' ').title()}: {score}")
        
        # Key metrics
        print(f"\\n📈 Key Metrics:")
        
        # Code quality
        quality_metrics = self.metrics.get('quality_gates', {}).get('code_quality_metrics', {})
        if 'ruff_issues' in quality_metrics:
            print(f"  • Code Issues: {quality_metrics['ruff_issues']}")
        
        # Testing
        testing_metrics = self.metrics.get('quality_gates', {}).get('testing_metrics', {})
        if 'line_coverage' in testing_metrics:
            print(f"  • Test Coverage: {testing_metrics['line_coverage']:.1f}%")
        
        # Security
        security_metrics = self.metrics.get('quality_gates', {}).get('security_metrics', {})
        if 'bandit_issues' in security_metrics:
            print(f"  • Security Issues: {security_metrics['bandit_issues']}")
        
        # Dependencies
        dep_metrics = self.metrics.get('dependencies', {})
        if 'total_dependencies' in dep_metrics:
            print(f"  • Dependencies: {dep_metrics['total_dependencies']}")
        
        # Automation
        cicd_metrics = self.metrics.get('automation_status', {}).get('cicd_metrics', {})
        if 'workflow_count' in cicd_metrics:
            print(f"  • CI/CD Workflows: {cicd_metrics['workflow_count']}")


def main():
    """Main function to collect SDLC metrics."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SDLC Metrics Collector')
    parser.add_argument('--output', '-o', type=Path, help='Output file path')
    parser.add_argument('--repo-root', '-r', type=Path, help='Repository root path')
    parser.add_argument('--config', '-c', type=Path, help='Metrics configuration file path')
    parser.add_argument('--json-only', action='store_true', help='Output JSON only')
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = SDLCMetricsCollector(args.repo_root, args.config)
    
    # Collect metrics
    metrics = collector.collect_all_metrics()
    
    # Save metrics
    if args.output:
        collector.save_metrics(args.output)
    else:
        collector.save_metrics()
    
    # Print summary unless JSON-only mode
    if not args.json_only:
        collector.print_summary()
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()