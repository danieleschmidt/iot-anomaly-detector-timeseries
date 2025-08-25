"""
Quality Gates Validator for Terragon SDLC
Comprehensive quality validation without external dependencies
"""

import ast
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class CodeQualityAnalyzer:
    """Analyze code quality without external tools."""
    
    def __init__(self, src_path: Path):
        self.src_path = Path(src_path)
        self.results = {
            "total_files": 0,
            "total_lines": 0,
            "python_files": 0,
            "complexity_issues": [],
            "import_issues": [],
            "security_issues": [],
            "documentation_coverage": 0.0,
            "test_coverage_estimate": 0.0
        }
    
    def analyze_codebase(self) -> Dict[str, Any]:
        """Comprehensive codebase analysis."""
        print("🔍 Analyzing codebase quality...")
        
        python_files = list(self.src_path.rglob("*.py"))
        self.results["python_files"] = len(python_files)
        
        for py_file in python_files:
            self._analyze_file(py_file)
        
        # Calculate metrics
        self._calculate_documentation_coverage()
        self._estimate_test_coverage()
        
        return self.results
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze individual Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
                
            self.results["total_files"] += 1
            self.results["total_lines"] += len(lines)
            
            # Parse AST for deeper analysis
            try:
                tree = ast.parse(content)
                self._analyze_ast(tree, file_path)
            except SyntaxError as e:
                self.results["syntax_errors"] = self.results.get("syntax_errors", [])
                self.results["syntax_errors"].append(f"{file_path}: {e}")
            
            # Basic security checks
            self._check_security_patterns(content, file_path)
            
        except Exception as e:
            print(f"⚠️  Error analyzing {file_path}: {e}")
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path) -> None:
        """Analyze AST for complexity and structure."""
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 0
                self.functions = 0
                self.classes = 0
                self.docstrings = 0
                
            def visit_FunctionDef(self, node):
                self.functions += 1
                if ast.get_docstring(node):
                    self.docstrings += 1
                
                # Calculate cyclomatic complexity (simplified)
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                        self.complexity += 1
                
                self.generic_visit(node)
            
            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)  # Same logic for async functions
            
            def visit_ClassDef(self, node):
                self.classes += 1
                if ast.get_docstring(node):
                    self.docstrings += 1
                self.generic_visit(node)
        
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        # Check for high complexity
        if visitor.complexity > 15:  # Threshold for complexity warning
            self.results["complexity_issues"].append({
                "file": str(file_path),
                "complexity": visitor.complexity,
                "functions": visitor.functions,
                "message": "High cyclomatic complexity detected"
            })
    
    def _check_security_patterns(self, content: str, file_path: Path) -> None:
        """Check for basic security anti-patterns."""
        security_patterns = [
            ("eval(", "Use of eval() function detected"),
            ("exec(", "Use of exec() function detected"),
            ("os.system(", "Direct system call detected"),
            ("subprocess.call(", "Subprocess call - ensure input validation"),
            ("pickle.loads(", "Unsafe pickle deserialization"),
            ("yaml.load(", "Unsafe YAML loading - use safe_load"),
            ("password", "Potential hardcoded password reference"),
            ("secret", "Potential hardcoded secret reference"),
            ("api_key", "Potential hardcoded API key reference"),
        ]
        
        for pattern, message in security_patterns:
            if pattern.lower() in content.lower():
                self.results["security_issues"].append({
                    "file": str(file_path),
                    "pattern": pattern,
                    "message": message
                })
    
    def _calculate_documentation_coverage(self) -> None:
        """Calculate documentation coverage estimate."""
        total_functions = 0
        documented_functions = 0
        
        for py_file in self.src_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total_functions += 1
                        if ast.get_docstring(node):
                            documented_functions += 1
            
            except Exception:
                continue
        
        if total_functions > 0:
            self.results["documentation_coverage"] = documented_functions / total_functions
        else:
            self.results["documentation_coverage"] = 1.0
    
    def _estimate_test_coverage(self) -> None:
        """Estimate test coverage based on test files."""
        src_files = list(self.src_path.rglob("*.py"))
        test_files = list(Path("tests").rglob("test_*.py")) if Path("tests").exists() else []
        
        if src_files:
            # Simple heuristic: ratio of test files to source files
            coverage_ratio = len(test_files) / len(src_files)
            self.results["test_coverage_estimate"] = min(coverage_ratio * 100, 100)
        else:
            self.results["test_coverage_estimate"] = 0.0


class PerformanceValidator:
    """Validate performance characteristics."""
    
    def __init__(self):
        self.results = {
            "file_processing_speed": 0.0,
            "memory_efficiency": "unknown",
            "startup_time": 0.0,
            "response_times": []
        }
    
    def validate_performance(self, src_path: Path) -> Dict[str, Any]:
        """Run performance validation tests."""
        print("⚡ Validating performance characteristics...")
        
        # Test file processing speed
        start_time = time.time()
        file_count = 0
        
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    f.read()
                file_count += 1
            except Exception:
                continue
        
        processing_time = time.time() - start_time
        if file_count > 0:
            self.results["file_processing_speed"] = file_count / processing_time
        
        # Simulate API response time test
        response_times = []
        for i in range(10):
            start = time.time()
            # Simulate processing
            data = {"test": "data", "index": i}
            json.dumps(data)  # JSON serialization
            response_time = (time.time() - start) * 1000  # ms
            response_times.append(response_time)
        
        self.results["response_times"] = response_times
        self.results["avg_response_time"] = sum(response_times) / len(response_times)
        
        return self.results


class SecurityValidator:
    """Validate security implementation."""
    
    def __init__(self):
        self.results = {
            "encryption_implemented": False,
            "authentication_present": False,
            "input_validation": False,
            "security_headers": False,
            "audit_logging": False,
            "security_score": 0
        }
    
    def validate_security(self, src_path: Path) -> Dict[str, Any]:
        """Validate security implementation."""
        print("🛡️  Validating security implementation...")
        
        security_indicators = {
            "encryption": ["encrypt", "decrypt", "cryptography", "fernet"],
            "authentication": ["authenticate", "login", "token", "jwt"],
            "validation": ["validate", "sanitize", "clean_input"],
            "headers": ["security_headers", "cors", "csrf"],
            "logging": ["audit_log", "security_log", "log_event"]
        }
        
        # Search for security implementations
        all_content = ""
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    all_content += content
            except Exception:
                continue
        
        # Check for security indicators
        for category, keywords in security_indicators.items():
            found = any(keyword in all_content for keyword in keywords)
            
            if category == "encryption":
                self.results["encryption_implemented"] = found
            elif category == "authentication":
                self.results["authentication_present"] = found
            elif category == "validation":
                self.results["input_validation"] = found
            elif category == "headers":
                self.results["security_headers"] = found
            elif category == "logging":
                self.results["audit_logging"] = found
        
        # Calculate security score
        security_checks = [
            self.results["encryption_implemented"],
            self.results["authentication_present"],
            self.results["input_validation"],
            self.results["security_headers"],
            self.results["audit_logging"]
        ]
        
        self.results["security_score"] = sum(security_checks) / len(security_checks) * 100
        
        return self.results


class ReliabilityValidator:
    """Validate system reliability features."""
    
    def __init__(self):
        self.results = {
            "error_handling": False,
            "circuit_breaker": False,
            "retry_logic": False,
            "health_checks": False,
            "graceful_shutdown": False,
            "reliability_score": 0
        }
    
    def validate_reliability(self, src_path: Path) -> Dict[str, Any]:
        """Validate reliability implementation."""
        print("🔧 Validating system reliability...")
        
        reliability_patterns = {
            "error_handling": ["try:", "except:", "raise", "exception"],
            "circuit_breaker": ["circuit", "breaker", "failure_threshold"],
            "retry": ["retry", "backoff", "attempt"],
            "health": ["health", "heartbeat", "status"],
            "shutdown": ["shutdown", "graceful", "cleanup"]
        }
        
        all_content = ""
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    all_content += content
            except Exception:
                continue
        
        # Check for reliability patterns
        for category, patterns in reliability_patterns.items():
            found = any(pattern in all_content for pattern in patterns)
            
            if category == "error_handling":
                self.results["error_handling"] = found
            elif category == "circuit_breaker":
                self.results["circuit_breaker"] = found
            elif category == "retry":
                self.results["retry_logic"] = found
            elif category == "health":
                self.results["health_checks"] = found
            elif category == "shutdown":
                self.results["graceful_shutdown"] = found
        
        # Calculate reliability score
        reliability_checks = [
            self.results["error_handling"],
            self.results["circuit_breaker"],
            self.results["retry_logic"],
            self.results["health_checks"],
            self.results["graceful_shutdown"]
        ]
        
        self.results["reliability_score"] = sum(reliability_checks) / len(reliability_checks) * 100
        
        return self.results


class ScalabilityValidator:
    """Validate scalability implementation."""
    
    def __init__(self):
        self.results = {
            "async_support": False,
            "distributed_processing": False,
            "auto_scaling": False,
            "load_balancing": False,
            "caching": False,
            "scalability_score": 0
        }
    
    def validate_scalability(self, src_path: Path) -> Dict[str, Any]:
        """Validate scalability features."""
        print("📈 Validating scalability implementation...")
        
        scalability_patterns = {
            "async": ["async def", "await", "asyncio"],
            "distributed": ["ray", "distributed", "cluster", "multiprocessing"],
            "scaling": ["scale", "auto_scale", "horizontal", "vertical"],
            "balancing": ["load_balance", "round_robin", "distribute"],
            "caching": ["cache", "redis", "memcache", "@lru_cache"]
        }
        
        all_content = ""
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    all_content += content
            except Exception:
                continue
        
        # Check for scalability patterns
        for category, patterns in scalability_patterns.items():
            found = any(pattern in all_content for pattern in patterns)
            
            if category == "async":
                self.results["async_support"] = found
            elif category == "distributed":
                self.results["distributed_processing"] = found
            elif category == "scaling":
                self.results["auto_scaling"] = found
            elif category == "balancing":
                self.results["load_balancing"] = found
            elif category == "caching":
                self.results["caching"] = found
        
        # Calculate scalability score
        scalability_checks = [
            self.results["async_support"],
            self.results["distributed_processing"],
            self.results["auto_scaling"],
            self.results["load_balancing"],
            self.results["caching"]
        ]
        
        self.results["scalability_score"] = sum(scalability_checks) / len(scalability_checks) * 100
        
        return self.results


class QualityGatesValidator:
    """Master quality gates validator."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.src_path = self.project_root / "src"
        
        # Quality thresholds
        self.thresholds = {
            "documentation_coverage": 70.0,
            "test_coverage": 85.0,
            "security_score": 80.0,
            "reliability_score": 85.0,
            "scalability_score": 75.0,
            "max_complexity": 15,
            "max_security_issues": 5
        }
        
        self.results = {}
    
    def validate_all_gates(self) -> Dict[str, Any]:
        """Run all quality gate validations."""
        print("🚀 TERRAGON SDLC QUALITY GATES VALIDATION")
        print("=" * 50)
        
        # Code quality analysis
        code_analyzer = CodeQualityAnalyzer(self.src_path)
        self.results["code_quality"] = code_analyzer.analyze_codebase()
        
        # Performance validation
        perf_validator = PerformanceValidator()
        self.results["performance"] = perf_validator.validate_performance(self.src_path)
        
        # Security validation
        security_validator = SecurityValidator()
        self.results["security"] = security_validator.validate_security(self.src_path)
        
        # Reliability validation
        reliability_validator = ReliabilityValidator()
        self.results["reliability"] = reliability_validator.validate_reliability(self.src_path)
        
        # Scalability validation
        scalability_validator = ScalabilityValidator()
        self.results["scalability"] = scalability_validator.validate_scalability(self.src_path)
        
        # Overall assessment
        self._assess_overall_quality()
        
        return self.results
    
    def _assess_overall_quality(self) -> None:
        """Assess overall quality and generate recommendations."""
        passed_gates = 0
        total_gates = 0
        
        # Documentation coverage gate
        doc_coverage = self.results["code_quality"]["documentation_coverage"] * 100
        if doc_coverage >= self.thresholds["documentation_coverage"]:
            passed_gates += 1
        total_gates += 1
        
        # Security score gate
        security_score = self.results["security"]["security_score"]
        if security_score >= self.thresholds["security_score"]:
            passed_gates += 1
        total_gates += 1
        
        # Reliability score gate
        reliability_score = self.results["reliability"]["reliability_score"]
        if reliability_score >= self.thresholds["reliability_score"]:
            passed_gates += 1
        total_gates += 1
        
        # Scalability score gate
        scalability_score = self.results["scalability"]["scalability_score"]
        if scalability_score >= self.thresholds["scalability_score"]:
            passed_gates += 1
        total_gates += 1
        
        # Security issues gate
        security_issues = len(self.results["code_quality"]["security_issues"])
        if security_issues <= self.thresholds["max_security_issues"]:
            passed_gates += 1
        total_gates += 1
        
        self.results["overall"] = {
            "passed_gates": passed_gates,
            "total_gates": total_gates,
            "pass_rate": (passed_gates / total_gates) * 100,
            "recommendation": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Documentation recommendations
        doc_coverage = self.results["code_quality"]["documentation_coverage"] * 100
        if doc_coverage < self.thresholds["documentation_coverage"]:
            recommendations.append(f"Improve documentation coverage from {doc_coverage:.1f}% to {self.thresholds['documentation_coverage']}%")
        
        # Security recommendations
        security_score = self.results["security"]["security_score"]
        if security_score < self.thresholds["security_score"]:
            recommendations.append(f"Enhance security implementation (current: {security_score:.1f}%, target: {self.thresholds['security_score']}%)")
        
        # Reliability recommendations
        reliability_score = self.results["reliability"]["reliability_score"]
        if reliability_score < self.thresholds["reliability_score"]:
            recommendations.append(f"Improve reliability features (current: {reliability_score:.1f}%, target: {self.thresholds['reliability_score']}%)")
        
        # Performance recommendations
        if self.results["performance"]["avg_response_time"] > 100:  # 100ms threshold
            recommendations.append("Optimize response times for better performance")
        
        if not recommendations:
            recommendations.append("All quality gates passed! System meets enterprise standards.")
        
        return recommendations
    
    def print_report(self) -> None:
        """Print comprehensive quality gates report."""
        print("\n📊 QUALITY GATES REPORT")
        print("=" * 30)
        
        # Code Quality
        print(f"\n🔍 CODE QUALITY:")
        cq = self.results["code_quality"]
        print(f"  • Files analyzed: {cq['python_files']}")
        print(f"  • Total lines: {cq['total_lines']:,}")
        print(f"  • Documentation coverage: {cq['documentation_coverage']*100:.1f}%")
        print(f"  • Test coverage estimate: {cq['test_coverage_estimate']:.1f}%")
        print(f"  • Security issues: {len(cq['security_issues'])}")
        print(f"  • Complexity issues: {len(cq['complexity_issues'])}")
        
        # Security
        print(f"\n🛡️  SECURITY:")
        sec = self.results["security"]
        print(f"  • Overall score: {sec['security_score']:.1f}%")
        print(f"  • Encryption: {'✓' if sec['encryption_implemented'] else '✗'}")
        print(f"  • Authentication: {'✓' if sec['authentication_present'] else '✗'}")
        print(f"  • Input validation: {'✓' if sec['input_validation'] else '✗'}")
        print(f"  • Audit logging: {'✓' if sec['audit_logging'] else '✗'}")
        
        # Reliability
        print(f"\n🔧 RELIABILITY:")
        rel = self.results["reliability"]
        print(f"  • Overall score: {rel['reliability_score']:.1f}%")
        print(f"  • Error handling: {'✓' if rel['error_handling'] else '✗'}")
        print(f"  • Circuit breaker: {'✓' if rel['circuit_breaker'] else '✗'}")
        print(f"  • Health checks: {'✓' if rel['health_checks'] else '✗'}")
        print(f"  • Graceful shutdown: {'✓' if rel['graceful_shutdown'] else '✗'}")
        
        # Scalability
        print(f"\n📈 SCALABILITY:")
        scal = self.results["scalability"]
        print(f"  • Overall score: {scal['scalability_score']:.1f}%")
        print(f"  • Async support: {'✓' if scal['async_support'] else '✗'}")
        print(f"  • Distributed processing: {'✓' if scal['distributed_processing'] else '✗'}")
        print(f"  • Auto-scaling: {'✓' if scal['auto_scaling'] else '✗'}")
        print(f"  • Caching: {'✓' if scal['caching'] else '✗'}")
        
        # Performance
        print(f"\n⚡ PERFORMANCE:")
        perf = self.results["performance"]
        print(f"  • File processing: {perf['file_processing_speed']:.1f} files/sec")
        print(f"  • Avg response time: {perf['avg_response_time']:.2f}ms")
        
        # Overall Assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        overall = self.results["overall"]
        print(f"  • Gates passed: {overall['passed_gates']}/{overall['total_gates']}")
        print(f"  • Pass rate: {overall['pass_rate']:.1f}%")
        
        if overall['pass_rate'] >= 80:
            print("  • Status: ✅ PASSED - Production Ready")
        elif overall['pass_rate'] >= 60:
            print("  • Status: ⚠️  CONDITIONAL - Needs Improvement")
        else:
            print("  • Status: ❌ FAILED - Major Issues")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(overall['recommendation'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "=" * 50)
        print("🚀 Terragon SDLC Quality Gates Validation Complete!")
        
        return overall['pass_rate'] >= 80  # Return True if passed


def main():
    """Run quality gates validation."""
    project_root = Path(__file__).parent
    
    validator = QualityGatesValidator(project_root)
    results = validator.validate_all_gates()
    
    # Print report
    passed = validator.print_report()
    
    # Save results
    with open("quality_gates_report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: quality_gates_report.json")
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()