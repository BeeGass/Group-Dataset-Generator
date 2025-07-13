# Contributing to Group Dataset Generator

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/s5-data-gen.git
   cd s5-data-gen
   ```

3. Install dependencies:
   ```bash
   uv pip install -e .
   ```

4. Run tests:
   ```bash
   uv run pytest tests/
   ```

## Development Workflow

### Adding a New Group Generator

1. Create a new file in `gdg/generators/`:
   ```python
   from ..base_generator import BaseGroupGenerator
   
   class MyGroupGenerator(BaseGroupGenerator):
       def get_group_name(self) -> str:
           return "mygroup"
       
       def get_elements(self, params: Dict) -> List[np.ndarray]:
           # Generate all group elements
           pass
   ```

2. Add corresponding test in `tests/generators/test_mygroup.py`

3. Update `gdg/generators/__init__.py` to export your generator

4. Add to `gdg/generate_all.py` if appropriate

### Code Style

- Use Black for formatting: `uv run black .`
- Type hints are encouraged
- Docstrings for all public methods
- Keep line length under 88 characters

### Testing

- Write tests for all new functionality
- Ensure mathematical correctness
- Test edge cases
- Run full test suite before submitting

### Pull Request Process

1. Create a feature branch:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Add my new feature"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/my-new-feature
   ```

4. Create a Pull Request with:
   - Clear description of changes
   - Any relevant issue numbers
   - Test results

## Adding New Group Families

When adding a new group family:

1. Research the mathematical properties
2. Implement efficient element generation
3. Verify group axioms in tests
4. Document the group's properties
5. Add examples to the documentation

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Documentation improvements
- General questions