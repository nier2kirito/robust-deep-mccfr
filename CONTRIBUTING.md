# Contributing to DL-MCCFR

Thank you for your interest in contributing to Deep Learning Monte Carlo Counterfactual Regret Minimization (DL-MCCFR)! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Development Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/DL_MCCFR.git
   cd DL_MCCFR
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests to verify setup**
   ```bash
   pytest tests/
   ```

## 🛠️ Development Workflow

### Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **pytest** for testing

Before submitting a pull request, run:

```bash
# Format code
black src/ tests/ examples/

# Check linting
flake8 src/ tests/ examples/

# Type checking
mypy src/

# Run tests
pytest tests/ -v
```

### Commit Messages

Use clear, descriptive commit messages:

- `feat: add new network architecture`
- `fix: resolve memory leak in training loop`
- `docs: update README with installation instructions`
- `test: add unit tests for feature extraction`

## 🎯 Areas for Contribution

### High Priority

1. **New Game Implementations**
   - Leduc Hold'em
   - Liar's Dice
   - Other imperfect information games

2. **Neural Network Architectures**
   - Graph Neural Networks
   - Recurrent architectures for sequential games
   - Attention mechanisms for large action spaces

3. **Performance Optimizations**
   - Multi-GPU training support
   - Distributed training
   - Memory optimization techniques

### Medium Priority

1. **Evaluation Metrics**
   - Additional exploitability measures
   - Strategy visualization tools
   - Convergence analysis

2. **Documentation**
   - Tutorial notebooks
   - API documentation improvements
   - Research paper implementations

3. **Testing**
   - Integration tests
   - Performance benchmarks
   - Edge case coverage

## 📝 Contribution Types

### Bug Reports

When reporting bugs, please include:

1. **Environment information**
   - Python version
   - PyTorch version
   - Operating system
   - Hardware (CPU/GPU)

2. **Reproduction steps**
   - Minimal code example
   - Expected behavior
   - Actual behavior
   - Error messages/stack traces

3. **Additional context**
   - Screenshots if applicable
   - Related issues or discussions

### Feature Requests

For new features, please provide:

1. **Problem description**
   - What problem does this solve?
   - Who would benefit from this feature?

2. **Proposed solution**
   - High-level design
   - API considerations
   - Performance implications

3. **Alternatives considered**
   - Other approaches you've considered
   - Why this approach is preferred

### Pull Requests

1. **Before you start**
   - Check existing issues and PRs
   - Discuss large changes in an issue first
   - Fork the repository

2. **Development process**
   - Create a feature branch from `main`
   - Write tests for new functionality
   - Ensure all tests pass
   - Update documentation as needed

3. **Pull request guidelines**
   - Provide clear description of changes
   - Reference related issues
   - Include test results
   - Request review from maintainers

## 🧪 Testing Guidelines

### Writing Tests

1. **Unit tests** for individual functions/classes
2. **Integration tests** for component interactions
3. **Performance tests** for critical paths
4. **Regression tests** for bug fixes

### Test Structure

```python
class TestNewFeature(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def test_basic_functionality(self):
        """Test basic feature behavior."""
        pass
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        pass
    
    def tearDown(self):
        """Clean up after tests."""
        pass
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_networks.py

# Run with coverage
pytest tests/ --cov=src/dl_mccfr --cov-report=html

# Run performance tests
pytest tests/ -m performance
```

## 📚 Documentation

### Code Documentation

- Use clear, descriptive docstrings
- Follow Google or NumPy docstring format
- Include type hints for all functions
- Document complex algorithms with references

### Example Docstring

```python
def train_network(model: nn.Module, data: torch.Tensor, 
                 epochs: int = 10) -> Dict[str, float]:
    """
    Train a neural network on the provided data.
    
    Args:
        model: PyTorch neural network model
        data: Training data tensor
        epochs: Number of training epochs
        
    Returns:
        Dictionary containing training metrics
        
    Raises:
        ValueError: If epochs is negative
        RuntimeError: If training fails
        
    Example:
        >>> model = create_network('simple', 27, 4)
        >>> data = torch.randn(100, 27)
        >>> metrics = train_network(model, data, epochs=5)
        >>> print(metrics['loss'])
    """
```

## 🔧 Adding New Games

To add a new game implementation:

1. **Create game module**
   ```
   src/dl_mccfr/games/your_game.py
   ```

2. **Implement required classes**
   - Game state representation
   - Action enumeration
   - Game logic (legal actions, terminal states, payoffs)
   - Feature extraction

3. **Add tests**
   ```
   tests/test_your_game.py
   ```

4. **Update documentation**
   - Add to `__init__.py`
   - Update README
   - Create example script

### Game Implementation Template

```python
from enum import Enum
from typing import List, Tuple

class YourGameAction(Enum):
    # Define game actions
    pass

class YourGameState:
    def __init__(self):
        # Initialize game state
        pass
    
    def is_terminal(self) -> bool:
        # Check if game is over
        pass
    
    def get_legal_actions(self) -> List[YourGameAction]:
        # Return legal actions
        pass
    
    def apply_action(self, action: YourGameAction) -> 'YourGameState':
        # Apply action and return new state
        pass
    
    def get_returns(self) -> List[float]:
        # Return payoffs for all players
        pass

class YourGame:
    def get_initial_state(self) -> YourGameState:
        # Return initial game state
        pass
```

## 🏆 Recognition

Contributors will be recognized in:

- README.md contributors section
- CHANGELOG.md for significant contributions
- Academic papers using this work (with permission)

## 📞 Getting Help

- **Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: For private inquiries

## 📄 License

By contributing to DL-MCCFR, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to DL-MCCFR! 🎉
