from .fetcher import get_aave_tvl
from .models import TVLPredictor
from .analysis import TVLAnalyzer
from .visualizer import TVLVisualizer

__version__ = '0.1.0'

__all__ = [
    'get_aave_tvl',
    'TVLPredictor',
    'TVLAnalyzer',
    'TVLVisualizer'
]