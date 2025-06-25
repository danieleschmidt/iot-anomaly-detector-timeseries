import json
import re
from pathlib import Path

plan = Path('DEVELOPMENT_PLAN.md').read_text()

# find first unchecked task
match = re.search(r"- \[ \] (.+)", plan)
if not match:
    raise SystemExit('No open tasks found')

epic = match.group(1).strip()

# basic decomposition; customizing for integration tests if appears
subtasks = []
if 'integration tests' in epic.lower():
    subtasks = [
        'Create integration test covering data generation, training, and detection pipeline',
        'Validate detection accuracy using labeled anomaly data',
        'Ensure integration test runs via `pytest` and passes',
        'Document integration test usage in README',
    ]
else:
    subtasks = [f'Implement {epic.lower()}']

# generate sprint board
board_lines = ['# Sprint Board', f'## Epic: {epic}', '']
for i, task in enumerate(subtasks, 1):
    board_lines.append(f'- [ ] {task}')
board = "\n".join(board_lines) + "\n"
Path('SPRINT_BOARD.md').write_text(board)

# acceptance criteria
criteria = {
    str(i): {
        'task': task,
        'acceptance': [
            f'{task} implemented',
            'All automated tests pass',
        ],
    }
    for i, task in enumerate(subtasks, 1)
}
Path('tests/sprint_acceptance_criteria.json').write_text(json.dumps(criteria, indent=2))
