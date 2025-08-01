#!/usr/bin/env python3
"""
Simplified Terragon Value Discovery - No External Dependencies
"""

import os
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

class SimpleValueDiscovery:
    def __init__(self, repo_path="/root/repo"):
        self.repo_path = Path(repo_path)
        
    def discover_value_items(self):
        items = []
        
        # 1. Git history hot spots
        items.extend(self._find_git_hotspots())
        
        # 2. TODO/FIXME comments
        items.extend(self._find_todo_comments())
        
        # 3. Large files that might need refactoring
        items.extend(self._find_large_files())
        
        # 4. Dependencies that might need updating
        items.extend(self._find_dependency_opportunities())
        
        # 5. Test coverage opportunities
        items.extend(self._find_test_opportunities())
        
        return items
    
    def _find_git_hotspots(self):
        items = []
        try:
            result = subprocess.run([
                'git', 'log', '--pretty=format:', '--name-only', '--since=90.days.ago'
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                file_changes = {}
                for line in result.stdout.strip().split('\\n'):
                    if line.strip() and line.endswith('.py'):
                        file_changes[line] = file_changes.get(line, 0) + 1
                
                # Find files changed more than 10 times
                for file_path, count in file_changes.items():
                    if count > 10:
                        items.append({
                            'id': f'hotspot_{hash(file_path) % 10000}',
                            'title': f'Refactor frequently changed file: {Path(file_path).name}',
                            'description': f'File changed {count} times in 90 days - potential technical debt',
                            'category': 'technical_debt',
                            'effort': min(count * 0.5, 16),
                            'business_value': 6,
                            'time_criticality': 4,
                            'risk_reduction': 8,
                            'files': [file_path],
                            'score': self._calculate_score(6, 4, 8, min(count * 0.5, 16))
                        })
        except:
            pass
        
        return items
    
    def _find_todo_comments(self):
        items = []
        try:
            result = subprocess.run([
                'grep', '-r', '-n', '--include=*.py', 'TODO\\|FIXME\\|XXX\\|HACK', 'src/'
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                todo_files = {}
                for line in result.stdout.strip().split('\\n'):
                    if ':' in line:
                        file_path = line.split(':')[0]
                        todo_files[file_path] = todo_files.get(file_path, 0) + 1
                
                for file_path, count in todo_files.items():
                    if count >= 2:  # Multiple TODOs in same file
                        items.append({
                            'id': f'todos_{hash(file_path) % 10000}',
                            'title': f'Address TODO/FIXME comments in {Path(file_path).name}',
                            'description': f'{count} TODO/FIXME comments need resolution',
                            'category': 'technical_debt',
                            'effort': count * 0.5,
                            'business_value': 5,
                            'time_criticality': 3,
                            'risk_reduction': 6,
                            'files': [file_path],
                            'score': self._calculate_score(5, 3, 6, count * 0.5)
                        })
        except:
            pass
        
        return items
    
    def _find_large_files(self):
        items = []
        try:
            for py_file in self.repo_path.glob('src/**/*.py'):
                if py_file.is_file():
                    line_count = sum(1 for _ in open(py_file))
                    if line_count > 300:  # Large file
                        items.append({
                            'id': f'large_{hash(str(py_file)) % 10000}',
                            'title': f'Refactor large file: {py_file.name}',
                            'description': f'File has {line_count} lines - consider breaking into smaller modules',
                            'category': 'code_quality',
                            'effort': line_count / 100,  # 1 hour per 100 lines
                            'business_value': 7,
                            'time_criticality': 3,
                            'risk_reduction': 6,
                            'files': [str(py_file)],
                            'score': self._calculate_score(7, 3, 6, line_count / 100)
                        })
        except:
            pass
        
        return items
    
    def _find_dependency_opportunities(self):
        items = []
        req_files = list(self.repo_path.glob('requirements*.txt'))
        
        for req_file in req_files:
            try:
                with open(req_file, 'r') as f:
                    deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                
                if deps:
                    items.append({
                        'id': f'deps_{hash(str(req_file)) % 10000}',
                        'title': f'Update dependencies in {req_file.name}', 
                        'description': f'Review and update {len(deps)} dependencies for security and compatibility',
                        'category': 'dependencies',
                        'effort': 2.0,
                        'business_value': 6,
                        'time_criticality': 7,
                        'risk_reduction': 9,
                        'files': [str(req_file)],
                        'score': self._calculate_score(6, 7, 9, 2.0)
                    })
            except:
                continue
        
        return items
    
    def _find_test_opportunities(self):
        items = []
        
        # Find Python files in src/ that might not have corresponding tests
        src_files = list(self.repo_path.glob('src/**/*.py'))
        test_files = set(f.name.replace('test_', '').replace('.py', '') for f in self.repo_path.glob('tests/test_*.py'))
        
        for src_file in src_files:
            if src_file.name != '__init__.py':
                base_name = src_file.name.replace('.py', '')
                if base_name not in test_files:
                    items.append({
                        'id': f'test_{hash(str(src_file)) % 10000}',
                        'title': f'Add tests for {src_file.name}',
                        'description': f'No corresponding test file found for {src_file.name}',
                        'category': 'testing',
                        'effort': 3.0,
                        'business_value': 7,
                        'time_criticality': 4,
                        'risk_reduction': 8,
                        'files': [str(src_file)],
                        'score': self._calculate_score(7, 4, 8, 3.0)
                    })
        
        return items
    
    def _calculate_score(self, business_value, time_criticality, risk_reduction, effort):
        """Calculate WSJF-like score"""
        cost_of_delay = business_value + time_criticality + risk_reduction
        return cost_of_delay / max(effort, 0.5)
    
    def generate_backlog_report(self):
        items = self.discover_value_items()
        
        # Sort by score descending
        items.sort(key=lambda x: x['score'], reverse=True)
        
        report = f"""# 📊 Terragon Autonomous Value Backlog

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Repository: iot-anomaly-detector-timeseries
Total Items Discovered: {len(items)}

## 🎯 Next Best Value Items

"""
        
        for i, item in enumerate(items[:10], 1):
            report += f"""### {i}. {item['title']} (Score: {item['score']:.2f})
- **Category**: {item['category'].title()}
- **Effort**: {item['effort']:.1f} hours
- **Business Value**: {item['business_value']}/10
- **Time Criticality**: {item['time_criticality']}/10
- **Risk Reduction**: {item['risk_reduction']}/10
- **Description**: {item['description']}
- **Files**: {', '.join(Path(f).name for f in item['files'])}

"""
        
        # Category breakdown
        categories = {}
        for item in items:
            categories[item['category']] = categories.get(item['category'], 0) + 1
        
        report += """## 📊 Category Breakdown

"""
        
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(items)) * 100 if items else 0
            report += f"- **{category.title()}**: {count} items ({percentage:.1f}%)\\n"
        
        # High-level metrics
        avg_score = sum(item['score'] for item in items) / len(items) if items else 0
        high_value_items = len([item for item in items if item['score'] > 5])
        
        report += f"""
## 📈 Value Metrics

- **Average Score**: {avg_score:.2f}
- **High-Value Items (>5.0)**: {high_value_items}
- **Total Estimated Effort**: {sum(item['effort'] for item in items):.1f} hours
- **Potential Value Delivery**: {len(items)} improvements identified

## 🔄 Discovery Sources

- ✅ Git History Analysis (Hot Spots)
- ✅ Code Comment Analysis (TODO/FIXME)
- ✅ File Size Analysis (Refactoring Opportunities)  
- ✅ Dependency Analysis (Updates Needed)
- ✅ Test Coverage Analysis (Missing Tests)

## 🎯 Recommendations

1. **Start with highest-scoring items** - Focus on technical debt and dependency updates
2. **Address security items immediately** - Any security-related findings should be prioritized
3. **Balance effort vs impact** - Mix quick wins with longer-term improvements
4. **Regular discovery cycles** - Re-run analysis weekly to identify new opportunities

---
*Generated by Terragon Autonomous SDLC Value Discovery Engine*
*Repository Maturity: Advanced (85%) → Target: Expert (95%)*
"""

        return report, items

if __name__ == "__main__":
    discovery = SimpleValueDiscovery()
    report, items = discovery.generate_backlog_report()
    
    # Save to file
    output_path = Path("/root/repo/.terragon/AUTONOMOUS_BACKLOG.md")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    # Also save items as JSON
    json_path = Path("/root/repo/.terragon/value-items.json")
    with open(json_path, 'w') as f:
        json.dump(items, f, indent=2)
    
    print(f"Discovery complete: {len(items)} items found")
    print(f"Report saved to: {output_path}")
    print(f"Items data saved to: {json_path}")
    
    # Show top 3 items
    if items:
        print("\\nTop 3 Value Items:")
        for i, item in enumerate(items[:3], 1):
            print(f"{i}. {item['title']} (Score: {item['score']:.2f})")