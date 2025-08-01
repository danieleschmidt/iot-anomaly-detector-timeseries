#!/usr/bin/env python3
"""
Repository Health Check Automation Script

This script performs comprehensive health checks on the repository
and generates reports for SDLC metrics tracking.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import yaml
import requests


class RepositoryHealthChecker:
    """Comprehensive repository health checker."""
    
    def __init__(self, repo_root: Path = None):
        self.repo_root = repo_root or Path.cwd()
        self.metrics = {}
        self.issues = []
        self.recommendations = []
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive report."""
        print("🔍 Starting repository health check...")
        
        # Core checks
        self.check_repository_structure()
        self.check_documentation_health()
        self.check_code_quality()
        self.check_security_configuration()
        self.check_ci_cd_configuration()
        self.check_dependency_health()
        self.check_testing_coverage()
        self.check_monitoring_setup()
        
        # Generate report
        report = self.generate_health_report()
        
        print("✅ Health check completed!")
        return report
    
    def check_repository_structure(self):
        """Check repository structure and organization."""
        print("📁 Checking repository structure...")
        
        required_files = [
            'README.md',
            'LICENSE',
            'requirements.txt',
            'pyproject.toml',
            '.gitignore',
            'Dockerfile',
            'docker-compose.yml',
            'Makefile'
        ]
        
        required_dirs = [
            'src/',
            'tests/',
            'docs/',
            '.github/',
            'config/'
        ]
        
        structure_score = 0
        total_items = len(required_files) + len(required_dirs)
        
        # Check files
        for file_path in required_files:
            if (self.repo_root / file_path).exists():
                structure_score += 1
            else:
                self.issues.append(f"Missing required file: {file_path}")
        
        # Check directories
        for dir_path in required_dirs:
            if (self.repo_root / dir_path).exists():
                structure_score += 1
            else:
                self.issues.append(f"Missing required directory: {dir_path}")
        
        self.metrics['repository_structure_score'] = (structure_score / total_items) * 100
    
    def check_documentation_health(self):
        """Check documentation completeness and quality."""
        print("📚 Checking documentation health...")
        
        doc_files = [
            'README.md',
            'CONTRIBUTING.md',
            'CODE_OF_CONDUCT.md',
            'SECURITY.md',
            'CHANGELOG.md',
            'docs/ARCHITECTURE.md',
            'docs/API_REFERENCE.md',
            'docs/DEPLOYMENT.md'
        ]
        
        existing_docs = 0
        total_docs = len(doc_files)
        
        for doc_file in doc_files:
            doc_path = self.repo_root / doc_file
            if doc_path.exists():
                existing_docs += 1
                # Check if documentation is substantial (> 100 characters)
                if doc_path.stat().st_size < 100:
                    self.issues.append(f"Documentation file too small: {doc_file}")
            else:
                self.issues.append(f"Missing documentation: {doc_file}")
        
        self.metrics['documentation_completeness'] = (existing_docs / total_docs) * 100
        
        # Check README quality
        readme_path = self.repo_root / 'README.md'
        if readme_path.exists():
            readme_content = readme_path.read_text().lower()
            readme_score = 0
            
            required_sections = [
                'installation', 'usage', 'contributing', 'license',
                'description', 'requirements', 'getting started'
            ]
            
            for section in required_sections:
                if section in readme_content:
                    readme_score += 1
            
            self.metrics['readme_quality_score'] = (readme_score / len(required_sections)) * 100
    
    def check_code_quality(self):
        """Check code quality metrics."""
        print("🔧 Checking code quality...")
        
        # Check if quality tools are configured
        quality_configs = [
            '.pre-commit-config.yaml',
            'pyproject.toml',
            '.editorconfig',
            'pytest.ini'
        ]
        
        config_score = 0
        for config_file in quality_configs:
            if (self.repo_root / config_file).exists():
                config_score += 1
            else:
                self.issues.append(f"Missing quality configuration: {config_file}")
        
        self.metrics['quality_config_score'] = (config_score / len(quality_configs)) * 100
        
        # Run code quality checks if tools are available
        try:
            # Check ruff
            result = subprocess.run(['ruff', 'check', '.'], 
                                  capture_output=True, text=True, cwd=self.repo_root)
            if result.returncode == 0:
                self.metrics['ruff_issues'] = 0
            else:
                issue_count = len(result.stdout.split('\\n')) if result.stdout else 0
                self.metrics['ruff_issues'] = issue_count
                if issue_count > 0:
                    self.issues.append(f"Ruff found {issue_count} issues")
        except FileNotFoundError:
            self.recommendations.append("Install ruff for code quality checking")
        
        # Check test coverage if pytest-cov is available
        try:
            result = subprocess.run(
                ['python', '-m', 'pytest', '--cov=src', '--cov-report=json'],
                capture_output=True, text=True, cwd=self.repo_root
            )
            
            coverage_file = self.repo_root / 'coverage.json'
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    self.metrics['test_coverage'] = coverage_data['totals']['percent_covered']
        except (FileNotFoundError, subprocess.CalledProcessError):
            self.recommendations.append("Run tests with coverage reporting")
    
    def check_security_configuration(self):
        """Check security configuration and practices."""
        print("🔒 Checking security configuration...")
        
        security_files = [
            'SECURITY.md',
            '.pre-commit-config.yaml',
            '.github/dependabot.yml'
        ]
        
        security_score = 0
        for sec_file in security_files:
            if (self.repo_root / sec_file).exists():
                security_score += 1
            else:
                self.issues.append(f"Missing security file: {sec_file}")
        
        # Check for security scanning in workflows
        workflows_dir = self.repo_root / '.github' / 'workflows'
        security_workflow = False
        
        if workflows_dir.exists():
            for workflow_file in workflows_dir.glob('*.yml'):
                content = workflow_file.read_text()
                if any(tool in content.lower() for tool in ['bandit', 'safety', 'semgrep', 'trivy']):
                    security_workflow = True
                    security_score += 1
                    break
        
        if not security_workflow:
            self.issues.append("No security scanning workflows found")
        
        self.metrics['security_config_score'] = (security_score / (len(security_files) + 1)) * 100
        
        # Check for common security issues
        gitignore_path = self.repo_root / '.gitignore'
        if gitignore_path.exists():
            gitignore_content = gitignore_path.read_text()
            security_patterns = ['.env', '*.key', '*.pem', 'secrets', 'credentials']
            
            missing_patterns = []
            for pattern in security_patterns:
                if pattern not in gitignore_content:
                    missing_patterns.append(pattern)
            
            if missing_patterns:
                self.recommendations.append(f"Add security patterns to .gitignore: {missing_patterns}")
    
    def check_ci_cd_configuration(self):
        """Check CI/CD pipeline configuration."""
        print("⚙️ Checking CI/CD configuration...")
        
        workflows_dir = self.repo_root / '.github' / 'workflows'
        cicd_score = 0
        
        if workflows_dir.exists():
            workflow_files = list(workflows_dir.glob('*.yml'))
            
            required_workflows = ['ci', 'test', 'build', 'deploy']
            found_workflows = []
            
            for workflow_file in workflow_files:
                content = workflow_file.read_text().lower()
                for workflow_type in required_workflows:
                    if workflow_type in workflow_file.name.lower() or workflow_type in content:
                        found_workflows.append(workflow_type)
                        break
            
            cicd_score = (len(set(found_workflows)) / len(required_workflows)) * 100
            self.metrics['cicd_completeness'] = cicd_score
            
            # Check for advanced features
            advanced_features = 0
            total_features = 4
            
            for workflow_file in workflow_files:
                content = workflow_file.read_text().lower()
                
                if 'matrix' in content:
                    advanced_features += 1
                if 'cache' in content:
                    advanced_features += 1
                if 'artifact' in content:
                    advanced_features += 1
                if any(env in content for env in ['staging', 'production']):
                    advanced_features += 1
                
                break  # Count features once across all workflows
            
            self.metrics['cicd_advanced_features'] = (advanced_features / total_features) * 100
        else:
            self.issues.append("No GitHub Actions workflows found")
            self.metrics['cicd_completeness'] = 0
    
    def check_dependency_health(self):
        """Check dependency management and security."""
        print("📦 Checking dependency health...")
        
        # Check requirements files
        req_files = [
            'requirements.txt',
            'requirements-dev.txt',
            'pyproject.toml'
        ]
        
        dependency_files = 0
        for req_file in req_files:
            if (self.repo_root / req_file).exists():
                dependency_files += 1
        
        if dependency_files == 0:
            self.issues.append("No dependency files found")
            self.metrics['dependency_management'] = 0
        else:
            self.metrics['dependency_management'] = (dependency_files / len(req_files)) * 100
        
        # Check for dependency scanning
        dependabot_config = self.repo_root / '.github' / 'dependabot.yml'
        if dependabot_config.exists():
            self.metrics['automated_dependency_updates'] = True
        else:
            self.issues.append("Dependabot configuration missing")
            self.metrics['automated_dependency_updates'] = False
        
        # Check for pinned versions
        req_path = self.repo_root / 'requirements.txt'
        if req_path.exists():
            requirements = req_path.read_text()
            lines = [line.strip() for line in requirements.split('\\n') if line.strip() and not line.startswith('#')]
            
            pinned_deps = 0
            for line in lines:
                if any(op in line for op in ['==', '>=', '<=', '~=', '!=']):
                    pinned_deps += 1
            
            if lines:
                self.metrics['dependency_pinning_ratio'] = (pinned_deps / len(lines)) * 100
            else:
                self.metrics['dependency_pinning_ratio'] = 0
    
    def check_testing_coverage(self):
        """Check testing setup and coverage."""
        print("🧪 Checking testing coverage...")
        
        tests_dir = self.repo_root / 'tests'
        if not tests_dir.exists():
            self.issues.append("No tests directory found")
            self.metrics['testing_setup'] = 0
            return
        
        # Count test files
        test_files = list(tests_dir.rglob('test_*.py'))
        src_files = list((self.repo_root / 'src').rglob('*.py')) if (self.repo_root / 'src').exists() else []
        
        if src_files:
            test_ratio = len(test_files) / len(src_files)
            self.metrics['test_to_source_ratio'] = min(test_ratio * 100, 100)  # Cap at 100%
        else:
            self.metrics['test_to_source_ratio'] = 0
        
        # Check for different test types
        test_types = {
            'unit': 0,
            'integration': 0,
            'e2e': 0,
            'performance': 0
        }
        
        for test_file in test_files:
            content = test_file.read_text().lower()
            if 'integration' in test_file.name.lower() or 'integration' in content:
                test_types['integration'] += 1
            elif 'e2e' in test_file.name.lower() or 'end_to_end' in test_file.name.lower():
                test_types['e2e'] += 1
            elif 'performance' in test_file.name.lower() or 'benchmark' in content:
                test_types['performance'] += 1
            else:
                test_types['unit'] += 1
        
        self.metrics['test_types_coverage'] = test_types
        
        # Check pytest configuration
        pytest_config = self.repo_root / 'pytest.ini'
        if pytest_config.exists():
            self.metrics['pytest_configured'] = True
        else:
            self.recommendations.append("Configure pytest.ini for better test management")
            self.metrics['pytest_configured'] = False
    
    def check_monitoring_setup(self):
        """Check monitoring and observability setup."""
        print("📊 Checking monitoring setup...")
        
        monitoring_files = [
            'config/monitoring/prometheus.yml',
            'config/grafana-dashboard.json',
            'docker-compose.yml'
        ]
        
        monitoring_score = 0
        for mon_file in monitoring_files:
            if (self.repo_root / mon_file).exists():
                monitoring_score += 1
        
        self.metrics['monitoring_setup'] = (monitoring_score / len(monitoring_files)) * 100
        
        # Check for health check endpoints
        src_files = list((self.repo_root / 'src').rglob('*.py')) if (self.repo_root / 'src').exists() else []
        health_check_found = False
        
        for src_file in src_files:
            try:
                content = src_file.read_text().lower()
                if 'health' in content and any(keyword in content for keyword in ['endpoint', 'route', 'api']):
                    health_check_found = True
                    break
            except UnicodeDecodeError:
                continue
        
        self.metrics['health_check_endpoint'] = health_check_found
        
        if not health_check_found:
            self.recommendations.append("Implement health check endpoints for monitoring")
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        
        # Calculate overall health score
        scores = [
            self.metrics.get('repository_structure_score', 0),
            self.metrics.get('documentation_completeness', 0),
            self.metrics.get('quality_config_score', 0),
            self.metrics.get('security_config_score', 0),
            self.metrics.get('cicd_completeness', 0),
            self.metrics.get('dependency_management', 0),
            self.metrics.get('test_to_source_ratio', 0),
            self.metrics.get('monitoring_setup', 0)
        ]
        
        overall_health = sum(scores) / len(scores)
        
        # Determine health status
        if overall_health >= 90:
            health_status = "Excellent"
            status_emoji = "🟢"
        elif overall_health >= 75:
            health_status = "Good"
            status_emoji = "🟡"
        elif overall_health >= 60:
            health_status = "Fair"
            status_emoji = "🟠"
        else:
            health_status = "Needs Improvement"
            status_emoji = "🔴"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_health_score': round(overall_health, 2),
            'health_status': health_status,
            'status_emoji': status_emoji,
            'metrics': self.metrics,
            'issues': self.issues,
            'recommendations': self.recommendations,
            'summary': {
                'total_issues': len(self.issues),
                'total_recommendations': len(self.recommendations),
                'critical_issues': [issue for issue in self.issues if any(keyword in issue.lower() 
                                  for keyword in ['security', 'missing', 'critical'])],
                'quick_wins': self.recommendations[:5]  # Top 5 recommendations
            }
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_path: Path = None):
        """Save health report to file."""
        if output_path is None:
            output_path = self.repo_root / 'repository_health_report.json'
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Health report saved to: {output_path}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print health report summary."""
        print(f"\\n{report['status_emoji']} Repository Health Report")
        print("=" * 50)
        print(f"Overall Health Score: {report['overall_health_score']}% ({report['health_status']})")
        print(f"Issues Found: {report['summary']['total_issues']}")
        print(f"Recommendations: {report['summary']['total_recommendations']}")
        
        if report['summary']['critical_issues']:
            print(f"\\n🚨 Critical Issues:")
            for issue in report['summary']['critical_issues']:
                print(f"  • {issue}")
        
        if report['summary']['quick_wins']:
            print(f"\\n💡 Quick Wins:")
            for rec in report['summary']['quick_wins']:
                print(f"  • {rec}")
        
        print(f"\\n📊 Key Metrics:")
        key_metrics = [
            ('Repository Structure', 'repository_structure_score'),
            ('Documentation', 'documentation_completeness'),
            ('Code Quality', 'quality_config_score'),
            ('Security', 'security_config_score'),
            ('CI/CD', 'cicd_completeness'),
            ('Testing', 'test_to_source_ratio'),
            ('Monitoring', 'monitoring_setup')
        ]
        
        for name, key in key_metrics:
            score = report['metrics'].get(key, 0)
            emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
            print(f"  {emoji} {name}: {score:.1f}%")


def main():
    """Main function to run repository health check."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Repository Health Checker')
    parser.add_argument('--output', '-o', type=Path, help='Output file path')
    parser.add_argument('--repo-root', '-r', type=Path, help='Repository root path')
    parser.add_argument('--json-only', action='store_true', help='Output JSON only')
    
    args = parser.parse_args()
    
    # Initialize checker
    checker = RepositoryHealthChecker(args.repo_root)
    
    # Run health check
    report = checker.run_all_checks()
    
    # Save report
    if args.output:
        checker.save_report(report, args.output)
    else:
        checker.save_report(report)
    
    # Print summary unless JSON-only mode
    if not args.json_only:
        checker.print_summary(report)
    else:
        print(json.dumps(report, indent=2))
    
    # Exit with appropriate code
    if report['overall_health_score'] < 60:
        sys.exit(1)  # Fail if health is poor
    elif report['summary']['critical_issues']:
        sys.exit(1)  # Fail if critical issues found
    else:
        sys.exit(0)  # Success


if __name__ == '__main__':
    main()